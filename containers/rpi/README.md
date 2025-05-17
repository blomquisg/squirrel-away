# Purpose

This is the bootc-enabled container for getting squirrel-away squirrel
detection installed on a Raspberry Pi 4.

# Before building

Before building this container, build the aarch64 libedgetpu.so and copy that
to `./var/lib/edgetpu/lib/`.  This is an important piece of the squirrel-away
squirrel detection, because it allows the AI inferencing code to hand off
inferencing to the Coral Edge TPU.  Without this, it's unlikely that the
Raspberry Pi can handle inferencing on its own.

# How to build

Build and push your container image to your favorite container repository
(e.g., quay.io).

```
$> podman build --build-arg "sshpubkey=ACTUAL_SSH_PUB_KEY" -t YOUR_IMAGE_TAG_NAME .
$> podman push YOUR_IMAGE_TAG_NAME
```

## Versioning hint

If you use `:latest` as the image tag version, your updates will be 
really simple because you won't have to specify a new version each time you
update.  This is really useful when hacking things together.  

It's a better practice to be explicit with the version numbers after you're 
past the hacking stage.

# Installing on Raspberry Pi

Part of the goal of this project was to create a very simple method of
deploying new changes to a Raspberry Pi.  I figured I'd be changing things
a lot in the beginning of this project, and I didn't want to have to keep
burning new images to a poor little SD Card.

So my approach was to start with a super simple bootc-enabled Fedora install,
then `bootc switch` to my desired container image.  I used Fedora IoT to get
the initial OS installed for two reasons.  First, Fedora IoT is known to work
with Raspberry Pis.  And two, Fedora IoT ships with bootc pre-installed.

## Getting bootc-enabled Fedora on a Raspberry Pi

I followed [these instructions](https://www.redhat.com/en/blog/fedora-iot-raspberry-pi) 
on the Red Hat Blog.  They were super simple, and got me exactly where where
I wanted to start.  But, stop before you unmount the SD Card and eject it (at
the end of the `Add WiFi networking to the image` section). Because I ran into
a couple of small issues that you may want to address before trying to run the
Raspberry Pi the first time.  You may not hit these issues.  If you don't,
that's great.  Move on to `Booting the Raspberry Pi with Fedora IoT` below.

My issues were zezere prevented the boot process from finishing, and the audit
logs were way too verbose, to the point where they ran all over the login 
prompt.

To fix these two issues, I did the following:

### Disable zezere

With the SD Card still mounted, mask the zezere service.

```
$> sudo ln -s /dev/null <sd_root_partition_mount>/etc/systemd/system/zezere_ignition.service
```

### Change the log level

I edited the `ostree-1.conf` file on the SD Card, which contained the grub
configurations.  For me, this file was located at `{sd_boot_partition_mount}/loader/entries/ostree-1.conf`.

I edited this file and changed the `options` line by adding the following at
the end:

```
loglevel=3
```

That was enough to quiet down the audit logs, so I could at least log in.


### Booting the Raspberry Pi with Fedora IoT

At this point you can finally unmount your SD Card (remember to unmount all of
the partitions, mine had three partitions).  And, then eject your SD Card from
the writer.

You can insert your SD Card into your Raspberry Pi and boot!  At this point, 
you should have Fedora IoT running.  If you used the same instructions from
the blog about configuring the root user with no password, you should just
be able to login as root.

## Switching to your container image

Remember, for this project, Fedora IoT is just a temporary starting point to
get a bootc-friendly OS in place.  The real goal is to use bootc to switch to
the Raspberry Pi container image you built and pushed above.

To switch to your bootable container image, just do the following once you've
logged into your Raspberry Pi:

```
$> bootc switch YOUR_IMAGE_NAME
$> reboot
```

When your Raspberry Pi reboots, you'll be running on your bootable container
image.


## Updating the OS Raspberry Pi

Once you get things running, you may need to do updates to your container
image.  This might include updating the squirrel-away squirrel detection code.
Or, maybe you want to add more files to your container image.  Whatever the 
changes you want to make, follow the instructions in the `How to build`
section above to build and push your updated container image.

Log into your Raspberry Pi.  Then run the following:

```
$> bootc upgrade
$> reboot
```

That's it.  You're done.  You've built your bootable container image.  You
have it running on a Raspberry Pi, and you have been able to push new changes
to your image and reload the OS on the Pi without having to burn a new image
on the SD Card.