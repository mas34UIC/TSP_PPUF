from Gen_TSP_Netlist_Class import  Gen_Netlists
from TSP_Gen_Class import TSP_Instance
from Run_Netlist_Class import Run_Netlists
from glob import glob
import os

'''
Generate, run, and modify netlists and graphs
'''

'''
Function to generate TSP instances
'''
def gen_instances(n, num_instances, dir, mean, var, metric, metric_broken):
    for i in range(1, num_instances+1):
        instance = TSP_Instance(n, mean, var)
        if metric:
            instance.gen_metric_TSP(using_coords=False)
        elif metric_broken:
            instance.gen_metric_TSP(using_coords=False)
            instance.break_triangle_inequality()
        else:
            instance.gen_non_metric_TSP()

        path = os.path.join(dir, f"{n}Node{i}", f"generated_{n}_node.csv")

        instance.save_TSP_to_csv(path)

def gen_netlists(paths):
    netlist_generator = Gen_Netlists()
    netlist_generator.gen_netlist(paths)

def run_netlists(n, dir):
    netlist_runner = Run_Netlists(n, dir)
    netlist_runner.analyze_all_netlists()

def main():
    nodes_low = 7
    nodes_high = 16
    mean = 15
    var = 1

    # generate and run instances in range [nodes_low, nodes_high]
    for i in range(nodes_low, nodes_high+1):

        ''' Define graph size and number of instances'''
        n = i
        num_instances = 1

        # root directory where tsp instances are contained
        dir = os.path.join(os.getcwd(), "IMCs", "Non-Metric2", f"{n}Node")

        ''' Generate TSP instances'''
        #gen_instances(n, num_instances, dir, mean, var, metric=False, metric_broken=True)

        ''' Generate netlists '''
        instance_paths = glob(os.path.join(dir, f"{n}Node*", f"generated_{n}_node.csv"))
        #gen_netlists(instance_paths)

        ''' Run netlists '''
        #run_netlists(n, dir)

        netlist_runner = Run_Netlists(n, dir)
        netlist_runner.plot_tour_length_histogram(os.path.join(os.getcwd(), "IMCs", "Non-Metric2", f"{n}Node"))

if __name__ == "__main__":
    main()