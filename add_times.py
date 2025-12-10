total = 0.0

with open(r"c:\Users\gpiga\Desktop\EK505_Final_Project\times.txt", "r") as f:
    for line in f:
        cleaned = line.strip()          # remove leading/trailing whitespace

        if not cleaned:
            continue                    # skip empty lines

        if cleaned.endswith(","):
            cleaned = cleaned[:-1]      # drop trailing comma

        total += float(cleaned)

print("Total time:", total)
