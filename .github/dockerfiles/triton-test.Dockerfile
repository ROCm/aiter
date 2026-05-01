FROM rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0

RUN pip config set global.default-timeout 60 && \
    pip config set global.retries 10

RUN pip uninstall -y triton pytorch-triton pytorch-triton-rocm triton-rocm amd-triton || true

COPY .github/requirements/triton-test.txt /tmp/triton-test.txt
COPY .github/scripts/verify_triton_pin.py /tmp/verify_triton_pin.py
RUN pip install -r /tmp/triton-test.txt
RUN python /tmp/verify_triton_pin.py /tmp/triton-test.txt
