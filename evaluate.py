from runners import ChallengeRunner

def main(args):

    scenario = 'assets/all_towns_traffic_scenarios.json'
    # scenario = 'assets/no_scenarios.json'


    #parser.add_argument('--agent', default='autoagents/image_agent')
    #parser.add_argument('--agent', default='autoagents/lbc_agent')
    #parser.add_argument('--agent', default='autoagents/cbs_agent')
    args.agent = 'autoagents/cbs2_agent'

    if(args.mod == 'fpn'):
        args.agent_config = 'results_lead/config_lead_fpn.yaml'
    elif(args.mod == 'ppm'):
        args.agent_config = 'results_lead/config_lead_ppm.yaml'
    else:
        args.agent_config = 'results_lead/config_lead_original.yaml'

    port = args.port
    tm_port = port + 2
    print('TEMPORARY: RUNNING ROUTES 10 to 20')
    for i in range(10,20):
        route = f'assets/routes_training/route_{i}.xml'
        #route = 'assets/routes_dev.xml'
        #route = 'assets/routes_training/route_10.xml'

        runner = ChallengeRunner(args, scenario, route, port=port, tm_port=tm_port)
        runner.run()
        del runner



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--timeout', default="60.0",
                        help='Set the CARLA client timeout value in seconds')

    parser.add_argument('--port', type=int, default=2000)

    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')
    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument("--mod", type=str,
                        default='original',
                        help="Perception module (original, ppm, fpn)")

    args = parser.parse_args()

    main(args)
