# Use the existing DeepVariant image as the base
FROM google/deepvariant:1.6.1-gpu
# Install unzip utility if not already present
RUN set -eux; \
    apt-get update && apt-get install -y unzip

# Debug: Check if the ZIP file exists
RUN set -eux; \
    ls -l /opt/deepvariant/bin/

# Ensure the correct permissions for the ZIP file
RUN set -eux; \
    chmod 644 /opt/deepvariant/bin/train.zip; \
    chmod 644 /opt/deepvariant/bin/make_examples.zip; \
    chmod 644 /opt/deepvariant/bin/call_variants.zip 

# Unzip the required file inside the container
RUN set -eux; \
    unzip -o /opt/deepvariant/bin/make_examples.zip -d /opt/deepvariant/bin/; \
    unzip -o /opt/deepvariant/bin/train.zip -d /opt/deepvariant/bin/; \
    unzip -o /opt/deepvariant/bin/call_variants.zip -d /opt/deepvariant/bin/

# Set PYTHONPATH to include the necessary directories
ENV PYTHONPATH="/opt/deepvariant/bin/runfiles/com_google_deepvariant:\
/opt/deepvariant/bin/runfiles/com_google_protobuf:\
/opt/deepvariant/bin/runfiles/six_archive:\
/opt/deepvariant/bin/runfiles/absl_py:$PYTHONPATH"

# Install Jupyter Lab and other dependencies
RUN set -eux; \
    apt-get update && \
    apt-get install -y python3-pip git zsh wget tmux && \
    pip3 install --no-cache-dir jupyterlab optuna


# Expose port for Jupyter Lab
EXPOSE 8888

# Set the default command to run Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
