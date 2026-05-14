FROM vastai/base-image:cuda-13.2-mini-py312-2026-03-18

MAINTAINER Sebastian Lueneburg <sebastian.lueneburg@gmail.com>

WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/trade

# Set environment variable
ENV PATH_TRADE=/app/trade

# RUN git clone --depth 1 my-calibration-repo .

# Copy everything from current project to /app
COPY .. /app
# Copy the ./folder directory to another location
COPY ./research /app/research

ENV PATH="/app:/app/research:${PATH}"

# Define the command to run your application
CMD [ "python", "./main.py" ]
