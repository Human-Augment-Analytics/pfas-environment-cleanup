import os
import subprocess
import sys

def run_pwscf_in_directory(directory):
    # Check if the provided directory exists
    if not os.path.isdir(directory):
        print("Error: Directory not found.")
        return
    
    # List all *.in files in the directory
    input_files = [f for f in os.listdir(directory) if f.endswith('.in')]
    
    # If no *.in files are found
    if not input_files:
        print(f"No '.in' files found in the directory '{directory}'.")
        return
    
    # Loop through each input file
    for input_file in input_files:
        input_path = os.path.join(directory, input_file)
        output_file = os.path.splitext(input_file)[0] + '.out'
        output_path = os.path.join(directory, output_file)
        
        print(f"Running pw.x on {input_file}...")
        
        try:
            # Run the pw.x calculation
            with open(output_path, 'w') as output_f:
                result = subprocess.run(['pw.x'], stdin=open(input_path), stdout=output_f, stderr=subprocess.PIPE)
                
            if result.returncode == 0:
                print(f"Successfully completed: {input_file} -> {output_file}")
            else:
                print(f"Error in calculation for: {input_file}")
                print(f"Error details: {result.stderr.decode()}")
        
        except Exception as e:
            print(f"Failed to execute pw.x for {input_file}: {e}")

def main():
    # Check if folder name is provided as argument
    if len(sys.argv) != 2:
        print("Usage: python run_pwscf.py <folder_name>")
        sys.exit(1)
    
    folder_name = sys.argv[1]
    run_pwscf_in_directory(folder_name)

if __name__ == '__main__':
    main()
