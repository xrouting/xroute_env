from gymnasium.envs.registration import register

register(
    id="xroute_env/ordering-training-v0",
    entry_point="xroute_env.envs:OrderingTrainingEnv",
)

# register(
#     id="xroute_env/ordering-evaluation-v0",
#     entry_point="xroute_env.envs:OrderingEvaluationEnv",
# )

# static_regions = [
#     {
#         "benchmark": "region1",
#         "from": "ISPD-2018 test1",
#         "size": "1x1",
#         "net": 36,
#         "pin": 30,
#         "sparsity": 1628.40,
#         "position": [(199500, 245100), (205200, 250800)]
#     }
# ]

# for region in static_regions:
#     register(
#         id="xroute_env/static-{}-v0".format(region['benchmark']),
#         entry_point="xroute_env.envs:StaticRegionEnv",
#         kwargs={"region": region},
#     )
