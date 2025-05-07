import yaml
import argparse

def load_yaml(file_path):
  with open(file_path, 'r') as f:
    return yaml.safe_load(f)

def sort_yaml_by_reference(reference_yaml, target_yaml):
  # Get the order of keys from reference yaml
  reference_keys = list(reference_yaml.keys())
  
  # Create a new ordered dictionary based on reference keys
  sorted_yaml = {}
  for key in reference_keys:
    if key in target_yaml:
      sorted_yaml[key] = target_yaml[key]
  
  # Add any remaining keys that weren't in reference
  for key in target_yaml:
    if key not in sorted_yaml:
      sorted_yaml[key] = target_yaml[key]
  
  return sorted_yaml

def main():
  # Get file paths from user input
  parser = argparse.ArgumentParser(description='Sort YAML file based on reference file')
  parser.add_argument('reference_file', help='Path to reference YAML file')
  parser.add_argument('target_file', help='Path to target YAML file to sort')
  args = parser.parse_args()
  reference_file = parser.parse_args().reference_file
  target_file = parser.parse_args().target_file

  # Load both YAML files
  reference_yaml = load_yaml(reference_file)
  target_yaml = load_yaml(target_file)

  # Sort target yaml based on reference yaml
  sorted_yaml = sort_yaml_by_reference(reference_yaml, target_yaml)

  # Print the sorted yaml
  print(yaml.dump(sorted_yaml, allow_unicode=True, sort_keys=False))

if __name__ == "__main__":
  main()