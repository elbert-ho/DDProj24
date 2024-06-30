from chembl_webresource_client.new_client import new_client

# Initialize the ChEMBL activity client
activity = new_client.activity

# Function to list available activity types
def list_activity_types():
    activities = activity.filter(limit=100)  # Get a sample of 1000 activities
    activity_types = set()

    count = 0
    for act in activities:
        if count >= 100:
            break
        count += 1
        if 'standard_type' in act:
            activity_types.add(act['standard_type'])

    return activity_types

# List available activity types
activity_types = list_activity_types()
print("Available activity types:", activity_types)
