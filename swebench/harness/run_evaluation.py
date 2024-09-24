from __future__ import annotations

import docker
import json
import resource# 导入 resource 库，用于设置和获取系统资源限制
import traceback# 导入 traceback 库，用于跟踪异常信息

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,# 导入常量，表示应用补丁失败
    APPLY_PATCH_PASS,# 导入常量，表示应用补丁成功
    INSTANCE_IMAGE_BUILD_DIR,# 导入常量，表示实例镜像构建目录
    KEY_INSTANCE_ID,# 导入常量，表示实例 ID 的键
    RUN_EVALUATION_LOG_DIR,# 导入常量，表示运行评估日志目录
)
from swebench.harness.docker_utils import (
    remove_image,# 导入函数，用于删除镜像
    copy_to_container,# 导入函数，用于将文件复制到容器中
    exec_run_with_timeout,# 导入函数，用于在超时时间内执行命令
    cleanup_container, # 导入函数，用于清理容器
    list_images,# 导入函数，用于列出镜像
    should_remove,# 导入函数，用于判断是否应删除镜像
    clean_images, # 导入函数，用于清理镜像
)
from swebench.harness.docker_build import (
    BuildImageError,# 导入类，用于处理构建镜像错误
    build_container,# 导入函数，用于构建容器
    build_env_images,# 导入函数，用于构建环境镜像
    close_logger,# 导入函数，用于关闭日志记录器
    setup_logger,# 导入函数，用于设置日志记录器
)
from swebench.harness.grading import get_eval_report# 导入函数，用于获取评估报告
from swebench.harness.test_spec import make_test_spec, TestSpec# 导入函数和类，用于创建测试规范和表示测试规范
from swebench.harness.utils import load_swebench_dataset, str2bool# 导入函数，用于加载数据集和字符串转布尔值


class EvaluationError(Exception):# 定义 EvaluationError 异常类，用于处理评估错误
    def __init__(self, instance_id, message, logger):# 初始化方法，接收实例 ID、消息和日志记录器
        super().__init__(message)# 调用父类的初始化方法
        self.super_str = super().__str__()# 获取父类的字符串表示
        self.instance_id = instance_id# 设置实例 ID
        self.log_file = logger.log_file# 设置日志文件
        self.logger = logger# 设置日志记录器

    def __str__(self):# 定义字符串表示方法
        return (
            f"Evaluation error for {self.instance_id}: {self.super_str}\n"
            f"Check ({self.log_file}) for more information."
        )# 返回错误信息，包括实例 ID 和日志文件路径


def run_instance(
        test_spec: TestSpec,# 测试规范实例
        pred: dict,# 预测字典，包括模型路径、补丁和实例 ID
        rm_image: bool,# 是否删除镜像
        force_rebuild: bool,# 是否强制重建镜像
        client: docker.DockerClient,# Docker 客户端
        run_id: str,# 运行 ID
        timeout: int | None = None,# 超时时间
    ):
    """
    Run a single instance with the given prediction.

    Args:
        test_spec (TestSpec): TestSpec instance
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
        rm_image (bool): Whether to remove the image after running
        force_rebuild (bool): Whether to force rebuild the image
        client (docker.DockerClient): Docker client
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    # Set up logging directory
    instance_id = test_spec.instance_id# 获取实例 ID
    model_name_or_path = pred.get("model_name_or_path", "None").replace("/", "__")# 获取模型路径，并替换路径分隔符
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id# 设置日志目录路径
    log_dir.mkdir(parents=True, exist_ok=True)# 创建日志目录

    # Link the image build dir in the log dir
    build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(":", "__")# 获取镜像构建目录
    image_build_link = log_dir / "image_build_dir"# 设置镜像构建链接路径
    if not image_build_link.exists():# 如果链接不存在
        try:
            # link the image build dir in the log dir
            image_build_link.symlink_to(build_dir, target_is_directory=True)# 创建符号链接
        except:
            # some error, idk why
            pass
    log_file = log_dir / "run_instance.log"# 设置日志文件路径

    # Set up report file + logger
    report_path = log_dir / "report.json"# 设置报告文件路径
    if report_path.exists():# 如果报告文件存在
        return instance_id, json.loads(report_path.read_text())# 返回实例 ID 和报告内容
    logger = setup_logger(instance_id, log_file)# 设置日志记录器

    # Run the instance
    container = None# 初始化容器变量
    try:
        # Build + start instance container (instance image should already be built)
        container = build_container(test_spec, client, run_id, logger, rm_image, force_rebuild)# 构建容器
        container.start()# 启动容器
        logger.info(f"Container for {instance_id} started: {container.id}")# 记录日志

        # Copy model prediction as patch file to container
        patch_file = Path(log_dir / "patch.diff")# 设置补丁文件路径
        patch_file.write_text(pred["model_patch"] or "")# 写入补丁内容
        logger.info(
            f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container..."
        )
        copy_to_container(container, patch_file, Path("/tmp/patch.diff"))# 将补丁文件复制到容器中

        # Attempt to apply patch to container
        val = container.exec_run(
            "git apply --allow-empty -v /tmp/patch.diff",# 在容器中执行命令，应用补丁
            workdir="/testbed",# 工作目录
            user="root",# 用户
        )
        if val.exit_code != 0:# 如果命令执行失败
            logger.info(f"Failed to apply patch to container, trying again...")
            
            # try "patch --batch --fuzz=5 -p1 -i {patch_path}" to try again
            val = container.exec_run(
                "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",# 使用patch命令应用补丁，允许5行模糊匹配
                workdir="/testbed",# 在/testbed目录下执行命令
                user="root",# 以root用户身份执行命令
            )
            if val.exit_code != 0:# 如果再次失败
                logger.info(f"{APPLY_PATCH_FAIL}:\n{val.output.decode('utf-8')}")
                raise EvaluationError(# 抛出评估错误
                    instance_id,# 传入实例ID
                    f"{APPLY_PATCH_FAIL}:\n{val.output.decode('utf-8')}", # 错误消息包含补丁应用失败的详细信息
                    logger,# 传入日志记录器
                )
            else:# 如果patch命令执行成功
                logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode('utf-8')}")# 记录成功信息到日志
        else:# 如果之前的git apply命令执行成功
            logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode('utf-8')}")# 记录成功信息到日志

        # Get git diff before running eval script
        git_diff_output_before = (
            container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()# 执行git diff命令并获取输出
        )
        logger.info(f"Git diff before:\n{git_diff_output_before}")# 记录运行前的git diff到日志

        eval_file = Path(log_dir / "eval.sh")# 创建评估脚本文件路径
        eval_file.write_text(test_spec.eval_script)# 将评估脚本内容写入文件
        logger.info(
            f"Eval script for {instance_id} written to {eval_file}; copying to container..."# 记录评估脚本已写入并正在复制到容器的信息
        )
        copy_to_container(container, eval_file, Path("/eval.sh"))# 将评估脚本复制到容器中

        # Run eval script, write output to logs
        test_output, timed_out, total_runtime = exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout)# 执行评估脚本，设置超时
        test_output_path = log_dir / "test_output.txt"# 创建测试输出文件路径
        logger.info(f'Test runtime: {total_runtime:_.2f} seconds')# 记录测试运行时间
        with open(test_output_path, "w") as f:# 打开测试输出文件
            f.write(test_output)# 写入测试输出
            logger.info(f"Test output for {instance_id} written to {test_output_path}")# 记录测试输出已写入文件的信息
            if timed_out:# 如果测试超时
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.") # 在输出文件中添加超时错误信息
                raise EvaluationError(# 抛出评估错误
                    instance_id,# 传入实例ID
                    f"Test timed out after {timeout} seconds.",# 错误消息包含超时信息
                    logger,# 传入日志记录器
                )

        # Get git diff after running eval script
        git_diff_output_after = (
            container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()# 执行git diff命令并获取输出
        )

        # Check if git diff changed after running eval script
        logger.info(f"Git diff after:\n{git_diff_output_after}")# 记录运行后的git diff到日志
        if git_diff_output_after != git_diff_output_before:# 如果git diff发生了变化
            logger.info(f"Git diff changed after running eval script")# 记录git diff变化的信息

        # Get report from test output
        logger.info(f"Grading answer for {instance_id}...")# 记录开始评分的信息
        report = get_eval_report(# 获取评估报告
            test_spec=test_spec,# 传入测试规范
            prediction=pred,# 传入预测结果
            log_path=test_output_path,# 传入测试输出路径
            include_tests_status=True,# 包含测试状态
        )
        logger.info(# 记录报告信息
            f"report: {report}\n"
            f"Result for {instance_id}: resolved: {report[instance_id]['resolved']}"
        )

        # Write report to report.json
        with open(report_path, "w") as f:# 打开报告文件
            f.write(json.dumps(report, indent=4))
        return instance_id, report# 返回实例ID和报告
    except EvaluationError as e:# 捕获评估错误
        error_msg = traceback.format_exc()# 获取错误堆栈
        logger.info(error_msg)# 记录错误信息
        print(e)# 打印错误
    except BuildImageError as e:# 捕获构建镜像错误
        error_msg = traceback.format_exc()# 获取错误堆栈
        logger.info(error_msg)# 记录错误信息
        print(e)
    except Exception as e:
        error_msg = (f"Error in evaluating model for {instance_id}: {e}\n"
                     f"{traceback.format_exc()}\n"
                     f"Check ({logger.log_file}) for more information.")
        logger.error(error_msg)
    finally:
        # Remove instance container + image, close logger
        cleanup_container(client, container, logger)
        if rm_image:
            remove_image(client, test_spec.instance_image_key, logger)
        close_logger(logger)
    return


def run_instances(
        predictions: dict,# 预测字典，包括所有实例的预测
        instances: list,# 实例列表
        cache_level: str,# 缓存级别
        clean: bool,# 是否清理镜像
        force_rebuild: bool,# 是否强制重建镜像
        max_workers: int,# 最大并行工作数
        run_id: str,# 运行 ID
        timeout: int,# 超时时间
    ):
    """
    Run all instances for the given predictions in parallel.

    Args:
        predictions (dict): Predictions dict generated by the model
        instances (list): List of instances
        cache_level (str): Cache level
        clean (bool): Clean images above cache level
        force_rebuild (bool): Force rebuild images
        max_workers (int): Maximum number of workers
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    client = docker.from_env()# 获取 Docker 客户端
    test_specs = list(map(make_test_spec, instances))# 创建测试规范列表

    # print number of existing instance images
    instance_image_ids = {x.instance_image_key for x in test_specs}# 获取实例镜像 ID 集合
    existing_images = {
        tag for i in client.images.list(all=True)
        for tag in i.tags if tag in instance_image_ids
    }# 获取现有镜像
    if not force_rebuild and len(existing_images):# 如果不强制重建且存在现有镜像
        print(f"Found {len(existing_images)} existing instance images. Will reuse them.")

    # run instances in parallel
    print(f"Running {len(instances)} instances...")
    with tqdm(total=len(instances), smoothing=0) as pbar:# 使用 tqdm 显示进度条
        with ThreadPoolExecutor(max_workers=max_workers) as executor:# 创建线程池
            # Create a future for running each instance
            futures = {
                executor.submit(
                    run_instance,
                    test_spec,
                    predictions[test_spec.instance_id],
                    should_remove(
                        test_spec.instance_image_key,
                        cache_level,
                        clean,
                        existing_images,
                    ),
                    force_rebuild,
                    client,
                    run_id,
                    timeout,
                ): None
                for test_spec in test_specs
            }# 为每个实例创建一个 future 任务
            # Wait for each future to complete
            for future in as_completed(futures):# 等待所有 future 任务完成
                pbar.update(1)# 更新进度条
                try:
                    # Update progress bar, check if instance ran successfully
                    future.result()# 获取 future 结果
                except Exception as e:
                    traceback.print_exc()# 打印异常信息
                    continue
    print("All instances run.")


def get_dataset_from_preds(
        dataset_name: str,# 数据集名称
        split: str,# 数据集分割
        instance_ids: list,# 实例 ID 列表
        predictions: dict,# 预测字典
        run_id: str,# 运行 ID
        exclude_completed: bool = True# 是否排除已完成的实例
    ):
    """
    Return only instances that have predictions and are in the dataset.
    If instance_ids is provided, only return instances with those IDs.
    If exclude_completed is True, only return instances that have not been run yet.
    """
    # load dataset
    dataset = load_swebench_dataset(dataset_name, split)# 加载数据集
    dataset_ids = {i[KEY_INSTANCE_ID] for i in dataset}# 获取数据集中的实例 ID 集合

    if instance_ids:
        # check that all instance IDs are in the dataset
        instance_ids = set(instance_ids)# 将实例 ID 列表转换为集合
        if instance_ids - dataset_ids:# 检查实例 ID 是否在数据集中
            raise ValueError(
                (
                    "Some instance IDs not found in dataset!"
                    f"\nMissing IDs:\n{' '.join(instance_ids - dataset_ids)}"
                )
            )
        # check that all instance IDs have predictions
        missing_preds = instance_ids - set(predictions.keys())# 检查实例 ID 是否有预测
        if missing_preds:
            print(f"Warning: Missing predictions for {len(missing_preds)} instance IDs.")
    
    # check that all prediction IDs are in the dataset
    prediction_ids = set(predictions.keys())# 获取预测中的实例 ID 集合
    if prediction_ids - dataset_ids:# 检查预测中的实例 ID 是否在数据集中
        raise ValueError(
            (
                "Some prediction IDs not found in dataset!"
                f"\nMissing IDs:\n{' '.join(prediction_ids - dataset_ids)}"
            )
        )

    if instance_ids:
        # filter dataset to just the instance IDs
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in instance_ids]# 过滤数据集，只保留指定的实例 ID7

    # check which instance IDs have already been run
    completed_ids = set()# 初始化已完成的实例 ID 集合
    for instance in dataset:# 遍历数据集中的实例
        if instance[KEY_INSTANCE_ID] not in prediction_ids:# 如果实例 ID 不在预测中
            # skip instances without predictions
            continue
        prediction = predictions[instance[KEY_INSTANCE_ID]]# 获取实例的预测
        report_file = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / prediction["model_name_or_path"].replace("/", "__")
            / prediction[KEY_INSTANCE_ID]
            / "report.json"
        )# 设置报告文件路径
        if report_file.exists():# 如果报告文件存在
            completed_ids.add(instance[KEY_INSTANCE_ID])# 添加实例 ID 到已完成集合

    if completed_ids and exclude_completed:# 如果有已完成的实例且需要排除已完成的实例
        # filter dataset to only instances that have not been run
        print(f"{len(completed_ids)} instances already run, skipping...")
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] not in completed_ids]# 过滤数据集，排除已完成的实例

    empty_patch_ids = {k for k, v in predictions.items() if v["model_patch"] == "" or v["model_patch"] is None}# 获取空补丁的实例 ID

    # filter dataset to only instances with predictions
    dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in prediction_ids and i[KEY_INSTANCE_ID] not in empty_patch_ids]# 过滤数据集，只保留有预测且补丁不为空的实例
    return dataset


def make_run_report(
        predictions: dict,
        full_dataset: list,
        client: docker.DockerClient,
        run_id: str
    ) -> Path:
    """
    Make a final evaluation and run report of the instances that have been run.
    Also reports on images and containers that may still running!

    Args:
        predictions (dict): Predictions dict generated by the model
        full_dataset (list): List of all instances
        client (docker.DockerClient): Docker client
        run_id (str): Run ID
    
    Returns:
        Path to report file
    """
    # instantiate sets to store IDs of different outcomes
    completed_ids = set()# 初始化已完成的实例 ID 集合
    resolved_ids = set()# 初始化已解决的实例 ID 集合
    error_ids = set()# 初始化错误的实例 ID 集合
    unstopped_containers = set()# 初始化未停止的容器集合
    unremoved_images = set()# 初始化未删除的镜像集合
    unresolved_ids = set()# 初始化未解决的实例 ID 集合
    incomplete_ids = set()# 初始化不完整的实例 ID 集合
    # get instances with empty patches
    empty_patch_ids = set()# 初始化空补丁的实例 ID 集合

    # iterate through dataset and check if the instance has been run
    for instance in full_dataset:# 遍历完整数据集中的实例
        instance_id = instance[KEY_INSTANCE_ID]# 获取实例 ID
        if instance_id not in predictions:# 如果实例 ID 不在预测中
            # skip instances without 
            incomplete_ids.add(instance_id)# 添加到不完整的实例 ID 集合
            continue
        prediction = predictions[instance_id]# 获取实例的预测
        if prediction.get("model_patch", None) in ["", None]:# 如果补丁为空
            empty_patch_ids.add(instance_id)# 添加到空补丁的实例 ID 集合
            continue
        report_file = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / prediction["model_name_or_path"].replace("/", "__")
            / prediction[KEY_INSTANCE_ID]
            / "report.json"
        )# 设置报告文件路径
        if report_file.exists():# 如果报告文件存在
            # If report file exists, then the instance has been run
            completed_ids.add(instance_id)# 添加到已完成的实例 ID 集合
            report = json.loads(report_file.read_text())# 读取报告文件内容
            if report[instance_id]["resolved"]:# 如果实例已解决
                # Record if the instance was resolved
                resolved_ids.add(instance_id)# 添加到已解决的实例 ID 集合
            else:
                unresolved_ids.add(instance_id)# 添加到未解决的实例 ID 集合
        else:
            # Otherwise, the instance was not run successfully
            error_ids.add(instance_id)# 添加到错误的实例 ID 集合

    # get remaining images and containers
    images = list_images(client)# 获取剩余的镜像
    test_specs = list(map(make_test_spec, full_dataset))# 创建测试规范列表
    for spec in test_specs:# 遍历测试规范
        image_name = spec.instance_image_key# 获取镜像名称
        if image_name in images:# 如果镜像在剩余的镜像中
            unremoved_images.add(image_name)# 添加到未删除的镜像集合
    containers = client.containers.list(all=True)# 列出所有容器
    for container in containers:# 遍历容器
        if run_id in container.name:# 如果运行 ID 在容器名称中
            unstopped_containers.add(container.name)# 添加到未停止的容器集合

    # print final report
    print(f"Total instances: {len(full_dataset)}")
    print(f"Instances submitted: {len(predictions)}")
    print(f"Instances completed: {len(completed_ids)}")
    print(f"Instances incomplete: {len(incomplete_ids)}")
    print(f"Instances resolved: {len(resolved_ids)}")
    print(f"Instances unresolved: {len(unresolved_ids)}")
    print(f"Instances with empty patches: {len(empty_patch_ids)}")
    print(f"Instances with errors: {len(error_ids)}")
    print(f"Unstopped containers: {len(unstopped_containers)}")
    print(f"Unremoved images: {len(unremoved_images)}")

    # write report to file
    report = {
        "total_instances": len(full_dataset),# 总实例数
        "submitted_instances": len(predictions),# 提交的实例数
        "completed_instances": len(completed_ids),# 完成的实例数
        "resolved_instances": len(resolved_ids),# 解决的实例数
        "unresolved_instances": len(unresolved_ids),# 未解决的实例数
        "empty_patch_instances": len(empty_patch_ids),# 空补丁的实例数
        "error_instances": len(error_ids),# 错误的实例数
        "unstopped_instances": len(unstopped_containers),# 未停止的容器数
        "completed_ids": list(sorted(completed_ids)),# 已完成的实例 ID 列表
        "incomplete_ids": list(sorted(incomplete_ids)),# 不完整的实例 ID 列表
        "empty_patch_ids": list(sorted(empty_patch_ids)),# 空补丁的实例 ID 列表
        "submitted_ids": list(sorted(predictions.keys())),# 提交的实例 ID 列表
        "resolved_ids": list(sorted(resolved_ids)),# 解决的实例 ID 列表
        "unresolved_ids": list(sorted(unresolved_ids)),# 未解决的实例 ID 列表
        "error_ids": list(sorted(error_ids)),# 错误的实例 ID 列表
        "unstopped_containers": list(sorted(unstopped_containers)),# 未停止的容器列表
        "unremoved_images": list(sorted(unremoved_images)),# 未删除的镜像列表
        "schema_version": 2,# 报告的架构版本
    }
    report_file = Path(
        list(predictions.values())[0]["model_name_or_path"].replace("/", "__")
        + f".{run_id}"
        + ".json"
    )# 设置报告文件路径
    with open(report_file, "w") as f:# 打开报告文件
        print(json.dumps(report, indent=4), file=f)
    print(f"Report written to {report_file}")
    return report_file


def get_gold_predictions(dataset_name: str, split: str):
    """
    Get gold predictions for the given dataset and split.
    """
    dataset = load_swebench_dataset(dataset_name, split)# 加载数据集
    return [
        {
            KEY_INSTANCE_ID: datum[KEY_INSTANCE_ID],
            "model_patch": datum["patch"],
            "model_name_or_path": "gold",
        } for datum in dataset
    ]# 返回金标准预测


def main(
        dataset_name: str,# 数据集名称
        split: str,# 数据集分割
        instance_ids: list,# 实例 ID 列表
        predictions_path: str,# 预测文件路径
        max_workers: int,# 最大并行工作数
        force_rebuild: bool,# 是否强制重建镜像
        cache_level: str,# 缓存级别
        clean: bool,# 是否清理镜像
        open_file_limit: int,# 打开文件限制
        run_id: str,# 运行 ID
        timeout: int,# 超时时间
    ):
    """
    Run evaluation harness for the given dataset and predictions.
    """
    # set open file limit
    assert len(run_id) > 0, "Run ID must be provided"
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))# 设置打开文件限制
    client = docker.from_env()# 获取 Docker 客户端

    # load predictions as map of instance_id to prediction
    if predictions_path == 'gold':# 如果使用金标准预测
        print("Using gold predictions - ignoring predictions_path")
        predictions = get_gold_predictions(dataset_name, split)# 获取金标准预测
    else:
        if predictions_path.endswith(".json"):# 如果预测文件是 JSON 格式
            with open(predictions_path, "r") as f:# 打开预测文件
                predictions = json.load(f)# 读取预测内容
        elif predictions_path.endswith(".jsonl"):# 如果预测文件是 JSONL 格式
            with open(predictions_path, "r") as f:# 打开预测文件
                predictions = [json.loads(line) for line in f]# 读取每行内容
        else:
            raise ValueError("Predictions path must be \"gold\", .json, or .jsonl")
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}# 创建预测字典

    # get dataset from predictions
    dataset = get_dataset_from_preds(dataset_name, split, instance_ids, predictions, run_id)# 从预测中获取数据集
    full_dataset = load_swebench_dataset(dataset_name, split)# 加载完整数据集
    existing_images = list_images(client)# 列出现有镜像
    print(f"Running {len(dataset)} unevaluated instances...")
    if not dataset:# 如果没有未评估的实例
        print("No instances to run.")
    else:
        # build environment images + run instances
        build_env_images(client, dataset, force_rebuild, max_workers)# 构建环境镜像
        run_instances(predictions, dataset, cache_level, clean, force_rebuild, max_workers, run_id, timeout)# 运行实例

    # clean images + make final report
    clean_images(client, existing_images, cache_level, clean)# 清理镜像
    make_run_report(predictions, full_dataset, client, run_id)# 生成最终报告


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", default="princeton-nlp/SWE-bench_Lite", type=str, help="Name of dataset or path to JSON file.")
    parser.add_argument("--split", type=str, default="test", help="Split of the dataset")
    parser.add_argument("--instance_ids", nargs="+", type=str, help="Instance IDs to run (space separated)")
    parser.add_argument("--predictions_path", type=str, help="Path to predictions file - if 'gold', uses gold predictions", required=True)
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of workers (should be <= 75%% of CPU cores)")
    parser.add_argument("--open_file_limit", type=int, default=4096, help="Open file limit")
    parser.add_argument(
        "--timeout", type=int, default=1_800, help="Timeout (in seconds) for running tests for each instance"
        )
    parser.add_argument(
        "--force_rebuild", type=str2bool, default=False, help="Force rebuild of all images"
    )
    parser.add_argument(
        "--cache_level",
        type=str,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="env",
    )
    # if clean is true then we remove all images that are above the cache level
    # if clean is false, we only remove images above the cache level if they don't already exist
    parser.add_argument(
        "--clean", type=str2bool, default=False, help="Clean images above cache level"
    )
    parser.add_argument("--run_id", type=str, required=True, help="Run ID - identifies the run")
    args = parser.parse_args()

    main(**vars(args))
