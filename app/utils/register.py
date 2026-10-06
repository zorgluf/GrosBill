
def get_environment(env_name):
    try:
        if env_name in ('frouge'):
            from environments.frouge.envs.frouge import FlammeRougeEnv
            return FlammeRougeEnv
        elif env_name in ('stotten'):
            from environments.stotten.envs.stotten import SchottenTottenEnv
            return SchottenTottenEnv
        elif env_name == 'stottentr':
            # same game as stotten, transformer policy + separate zoo/logs namespace
            from environments.stotten.envs.stotten import SchottenTottenTrEnv
            return SchottenTottenTrEnv
        elif env_name == 'smallw':
            from environments.smallw.envs.smallw import SmallWorldEnv
            return SmallWorldEnv
        elif env_name in ('smallw2', 'smallw4', 'smallw5'):
            # same game on the 2/4/5-player board: own spaces, zoo and logs
            from environments.smallw.envs import smallw
            return {'smallw2': smallw.SmallWorld2Env, 'smallw4': smallw.SmallWorld4Env,
                    'smallw5': smallw.SmallWorld5Env}[env_name]
        elif env_name == 'jamaica':
            # every game drawn among 3-6 players: one network for every count
            from environments.jamaica.envs.jamaica import JamaicaAllCountsEnv
            return JamaicaAllCountsEnv
        else:
            raise Exception(f'No environment found for {env_name}')
    except SyntaxError as e:
        print(e)
        raise Exception(f'Syntax Error for {env_name}!')
    except Exception as e:
        raise Exception(f'Install the environment first using: \nbash scripts/install_env.sh {env_name}\nAlso ensure the environment is added to /utils/register.py') from e
    


def get_network_arch(env_name):
    if env_name in ('frouge'):
        from models.frouge.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('stotten'):
        from models.stotten.models import CustomPolicy
        return CustomPolicy
    elif env_name == 'stottentr':
        from models.stotten.models import TransformerPolicy
        return TransformerPolicy
    elif env_name in ('smallw', 'smallw2', 'smallw4', 'smallw5'):
        # one network per player count, sized from the observation space
        from models.smallw.models import CustomPolicy
        return CustomPolicy
    elif env_name == 'jamaica':
        from models.jamaica.models import CustomPolicy
        return CustomPolicy
    else:
        raise Exception(f'No model architectures found for {env_name}')
