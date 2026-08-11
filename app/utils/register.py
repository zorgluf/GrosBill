import os

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
        else:
            raise Exception(f'No environment found for {env_name}')
    except SyntaxError as e:
        print(e)
        raise Exception(f'Syntax Error for {env_name}!')
    except:
        raise Exception(f'Install the environment first using: \nbash scripts/install_env.sh {env_name}\nAlso ensure the environment is added to /utils/register.py')
    


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
    else:
        raise Exception(f'No model architectures found for {env_name}')
    
def get_trajectory_path(env_name: str):
    dir = f"zoo/trajectories/{env_name}"
    if os.path.exists(dir) == False:
        os.makedirs(dir)
    return dir

