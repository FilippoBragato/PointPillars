from postproces import *
import argparse

def main(args):
    results = open_results(args.results)
    results = [change_format(result) for result in results]
    compute_average_precision(results)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--results', default='', help='your results file')
    
    args = parser.parse_args()

    main(args)