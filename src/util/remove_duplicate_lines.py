#!/usr/bin/env python3
import sys

def remove_duplicate_lines(input_file, output_file=None):
    seen = set()
    unique_lines = []

    # Read through each line in the file.
    with open(input_file, 'r') as f:
        for line in f:
            # If the line hasn't been seen before, add it to our results.
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)

    # If an output file is specified, write the unique lines to it.
    if output_file:
        with open(output_file, 'w') as out:
            out.writelines(unique_lines)
    else:
        # Otherwise, print the unique lines to standard output.
        for line in unique_lines:
            print(line, end='')

if __name__ == '__main__':
    # Check for correct usage.
    '''if len(sys.argv) < 2:
        print("Usage: {} input_file [output_file]".format(sys.argv[0]))
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None'''

    input_file = "../../data/output/counterexample_n12_violated_circular_duplicates.txt"
    output_file = "out/unsatisfied_circular/counterexample_n12_violated_circular.txt"

    remove_duplicate_lines(input_file, output_file)
