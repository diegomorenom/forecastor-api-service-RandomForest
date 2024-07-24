# Use the Amazon Linux base image for AWS Lambda
FROM public.ecr.aws/lambda/python:3.11

# Update and install necessary packages for building h5py and TensorFlow dependencies
RUN yum update -y && \
    yum install -y \
    gcc \
    gcc-c++ \
    make \
    pkgconfig \
    hdf5 \
    hdf5-devel \
    zlib-devel \
    && yum clean all

# Set up your AWS Lambda environment
WORKDIR /var/task

# Copy the requirements file
COPY requirements.txt .

# Install the function's dependencies
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}" -U --no-cache-dir

# Copy function code
COPY . ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (adjust as per your actual handler function)
CMD ["main.handler"]
