# Start from a PyTorch image with CUDA support
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# Set environment variables (optional)
# ENV TOKENIZERS_PARALLELISM=true

# Run the script.py when the container launches, adjust the command according to your script's name and expected arguments
CMD ["/bin/bash"]
