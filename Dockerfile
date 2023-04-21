# Use the latest image of nvcr.io/nvidia/pytorch as base
FROM nvcr.io/nvidia/pytorch:23.03-py3

# Set the working directory to /app
WORKDIR /build

COPY requirements.txt /build/

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip \
    && pip install --root-user-action=ignore --trusted-host pypi.python.org -r requirements.txt

