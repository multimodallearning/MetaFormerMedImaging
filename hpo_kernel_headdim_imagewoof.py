from clearml import Task
from clearml.automation import HyperParameterOptimizer, DiscreteParameterRange, GridSearch

task = Task.init(project_name="FlexConv/Classification/HPO", task_name='Kernel & HeadDim', auto_resource_monitoring=False,
                 reuse_last_task_id=False, auto_connect_frameworks=False)

controller = HyperParameterOptimizer(
    base_task_id='466b14a9f22a487483c50c6e0912c0bc',
    hyper_parameters=[
        DiscreteParameterRange('Args/fit.model.init_args.head_dim', values=[16, 32, 64]),
        DiscreteParameterRange('Args/fit.model.init_args.kernel_size', values=[3, 5, 7]),
        DiscreteParameterRange('Args/fit.model.init_args.learn_pe', values=[True, False]),
        DiscreteParameterRange('Args/fit.model.init_args.use_slopes', values=[True, False]),
    ],
    objective_metric_title='f1',
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
