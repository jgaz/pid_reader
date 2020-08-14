FROM python:3.8-buster
RUN pip install -y git+https://github.com/jgaz/pid_reader.git@master
# Read the environment variables needed
ENV AZURE_STORAGE_CONNECTION_STRING "$AZURE_STORAGE_CONNECTION_STRING"
ENV BLOB_PATH "$BLOB_PATH"
ENV NUMBER_DIAGRAMS "$NUMBER_DIAGRAMS"
ENV SYMBOLS_PER_DIAGRAM "$SYMBOLS_PER_DIAGRAM"
ENV DIAGRAM_MATTER "$DIAGRAM_MATTER"
ENV DIAGRAM_SIZE "$DIAGRAM_SIZE"

# Generate and push the training dataset
CMD ""
