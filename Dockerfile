FROM flatironinstitute/triqs:latest

# Create a working directory.
RUN mkdir triqs_spectrometer
WORKDIR triqs_spectrometer

ENV PATH=/home/triqs/.local/bin:${PATH}


# Install Python dependencies.
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the rest of the codebase into the image
COPY . ./

# Finally, run gunicorn.
# CMD [ "gunicorn", "--workers=4", "--threads=1", "-b 0.0.0.0:9375", "app:server"]
# or run in debug mode
CMD ["python3", "app.py"]