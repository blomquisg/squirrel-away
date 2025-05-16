## TODO list for Squirrel Away

- [ ] Write script to download and populate training data directories
- [x] Download appropriate CNN model for animal image recognition (MobileNetV2)
- [x] Work on code for fine-tuning CNN model to specialize recognizing squirrels vs. other animals
- [ ] Hardware stuff ... (there's a fair amount here that needs to be fleshed out and documented)
- [x] Model training/fine-tuning
- [ ] Enable Coral TPU with Fedora
- [ ] Convert model to work with Coral TPU
- [ ] Repeatable build process for RPi
- [ ] Deployment and live testing


Build Steps:

1. Get the container image to build.  Make sure it's in root's images because the bootc image build will need to run as root.
    `sudo podman pull quay.io/blomquisg/squirrel-away-fedora`
2. Get the bootc image builder image.  A matter of convenience really.
    `sudo podman pull quay.io/centos-bootc/bootc-image-builder:latest`
3. Prep the build environment.  Make sure you're in the directory where you expect to output build artifacts.
    `mkdir config output`
4. Build the raw image for Raspberry Pi.  Some qemu/bootc bugs were fixed in Fedora 41.
    `sudo podman run --rm   --privileged   --pull=newer   --security-opt label=type:unconfined_t   -v ./config/config.toml:/config.toml:ro   -v ./output:/output   -v /var/lib/containers/storage:/var/lib/containers/storage   quay.io/centos-bootc/bootc-image-builder:latest   --local   --type raw --rootfs ext4 --target-arch aarch64 quay.io/blomquisg/squirrel-away-fedora`
