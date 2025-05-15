from clearml import Task
from clearml.automation import HyperParameterOptimizer, DiscreteParameterRange, GridSearch

task = Task.init(project_name="FlexConv/Classification/HPO", task_name='Kernel & HeadDim', auto_resource_monitoring=False,
                 reuse_last_task_id=False, auto_connect_frameworks=False)

controller = HyperParameterOptimizer(
    base_task_id='c2c902ca0e354077ac5f792df12bf344',
    hyper_parameters=[
        DiscreteParameterRange('Args/fit.model.init_args.head_dim', values=[16, 32, 64]),
        DiscreteParameterRange('Args/fit.model.init_args.kernel_size', values=[3, 5, 7]),
    ],
    objective_metric_title='acc',
    objective_metric_series='val',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=1,
    optimizer_class=GridSearch,
)

controller.start_locally()
# wait until optimization completed or timed-out
controller.wait()
# make sure we stop all jobs
controller.stop()
