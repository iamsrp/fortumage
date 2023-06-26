#!/usr/bin/env python3

from   diffusers          import StableDiffusionPipeline
from   PIL.PngImagePlugin import PngInfo

import argh
import logging as LOG
import os
import random
import time
import torch

# ------------------------------------------------------------------------------

LOG.basicConfig(
    format='[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s',
    level=LOG.INFO
)

# ------------------------------------------------------------------------------

class Fortune():
    """
    A class which pulls out text from the fortune files and delivers it to the
    user.
    """
    def __init__(self,
                 fortunes_dir="/usr/share/games/fortunes"):
        """
        :param fortunes_dir:
            The location of the fortune data files.
        """
        self._dir = fortunes_dir


    def pick(self, max_length=1000):
        """
        Choose a random fortune. This is the meat of this class.

        :param max_length:
            The maximum length of a selected fortune, in bytes.
        """
        max_length = int(max_length)

        # We do this all from scratch each time since it's not _that_ expensive
        # and it means we don't have to restart anything when new files are
        # added. We have a list of filenames and the start and end of their data
        # as part of the total count.
        #
        # We are effectively concatenating the files here so as to avoid
        # bias. Consider: if you have two files, with one twice the size of the
        # other, if we picked a random fortune from a random file then then
        # fortunes in the smallee file would be twice as likely to come up as
        # ones in the bigger one.
        file_info = []
        total_size = 0
        for (subdir, _, files) in os.walk(self._dir, followlinks=True):
            for filename in files:
                # The fortune files have an associated .dat file, this means we
                # can identify them by looking for that .dat file.
                path = os.path.join(subdir, filename)
                dat_path = path + '.dat'
                LOG.debug("Candidate: %s %s", path, dat_path)
                if os.path.exists(dat_path):
                    # Open it to make sure can do so
                    try:
                        with open(path, 'rt'):
                            # Get the file length to use it to accumulate into
                            # our running counter, and to compute the file-
                            # specifc stats.
                            stat = os.stat(path)

                            # The start of the file is the current total_size
                            # and the end is that plus the file size
                            start = total_size
                            total_size += stat.st_size
                            end = total_size
                            file_info.append((path, start, end))
                            LOG.debug("Adding %s[%d:%d]", path, start, end)
                    except Exception as e:
                        LOG.debug("Failed to add %s: %s", path, e)


        # Keep trying this until we get something, or until we give up. Most of
        # the time we expect this to work on the first go unless something weird
        # is going on.
        for tries in range(10):
            LOG.debug("Try #%d", tries)

            # Now that we have a list of files, pick one at random by choosing a
            # point somewhere in there
            offset = random.randint(0, total_size)
            LOG.debug("Picked offset %d", offset)

            # Now we look for the file which contains that offset
            for (filename, start, end) in file_info:
                if start <= offset < end:
                    with open(filename, 'rt') as fh:
                        # Jump to the appropriate point in the file, according to
                        # the offset (relative to the files's start in the overall
                        # set)
                        seek_offset = offset - start
                        if seek_offset > 0:
                            fh.seek(seek_offset)

                        try:
                            # Now look for the bracketing '%'s. Read in a nice big
                            # chunk and hunt for it in there.
                            chunk = fh.read(min(10 * max_length, 1024 * 1024))

                            # The file could start with a bracketer and we want
                            # to catch that
                            if seek_offset == 0 and chunk.startswith('%\n'):
                                s = 2
                            else:
                                s = chunk.index('\n%\n') + 3

                            # Now look for the end. A properly-formed file
                            # should have a '%\n' as its last line.
                            e = chunk.index('\n%\n', s)

                            # We found a match. Is it small enough?
                            LOG.debug("Found section %s[%d:%d]", filename, s, e)
                            if (e - s) > max_length:
                                # Nope, go around and try again
                                break
                            else:
                                # Yes!
                                return chunk[s:e]

                        except ValueError:
                            # Find to match so give up and go around again
                            break

        # If we got here then we gave up trying
        return None


class ImageMaker():
    def __init__(self, model_id = "runwayml/stable-diffusion-v1-5"):
        """
        CTOR

        :param model_id: The model identifier string.
        """
        self._model_id = model_id
        self._pipe = StableDiffusionPipeline.from_pretrained(
                         self._model_id,
                         torch_dtype=torch.float16
                     )
        self._pipe = self._pipe.to("cuda")


    def make_and_save_image(self, prompt, filename):
        """
        Create a PNG image and save it with the given filename.

        :param prompt:   The prompt to use to generate the image.
        :param filename: The filename to save the image as.
        """
        # Make sure that the filename is a PNG one
        filename = str(filename)
        if not filename.endswith('.png'):
            raise ValueError("Filename must end with '.png', had: %s" % filename)

        # Create them, and choose the first one which looks non-empty. The NSFW
        # checker will blank out images which are dodgy.
        images = self._pipe(prompt).images
        image = None
        for i in images:
            if len([v for v in i.histogram() if v > 0]) > 10:
                image = i
                break

        # Save what we got
        if image:
            metadata = PngInfo()
            metadata.add_text("prompt", prompt)
            metadata.add_text("model_id", self._model_id)
            image.save(filename, pnginfo=metadata)
            return True
        else:
            return False

# ------------------------------------------------------------------------------

# Main entry point
@argh.arg('--fortunes_dir', '-i',
          help="The location of the fortune files")
@argh.arg('--outdir', '-o',
          help="Where to write the output image files")
@argh.arg('--sleep', '-s',
          help="How long to sleep between generations")
@argh.arg('--latest', '-l',
          help="Whether to create a 'latest' link")
def main(fortunes_dir='/usr/share/games/fortunes',
         outdir='.',
         sleep=200,
         latest=True):
    
    # Need these
    fortune = Fortune(fortunes_dir=fortunes_dir)
    maker   = ImageMaker()

    if latest:
        latest_fn = os.path.join(outdir, 'latest.png')
    else:
        latest_fn = None

    # Loop forever
    while True:
        prompt = fortune.pick()
        if prompt:
            # We can make it
            outname = '%d.png' % time.time()
            fn = os.path.join(outdir, outname)
            LOG.info("Generating %s from:\n%s", fn, prompt)
            if not maker.make_and_save_image(prompt, fn):
                LOG.warning("Failed to create an image, looping to try again")
                time.sleep(1)
                continue
            LOG.info("Done")

            # Create the latest link?
            if latest_fn:
                if os.path.exists(latest_fn):
                    LOG.info("Removing old %s link", latest_fn)
                    os.remove(latest_fn)
                LOG.info("Linking %s to %s", latest_fn, outname)
                os.symlink(outname, latest_fn)

            # And wait for a bit before making the next one
            time.sleep(sleep)
        else:
            LOG.error("No prompt!")
            time.sleep(1)

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        argh.dispatch_command(main)
    except Exception as e:
        print("%s" % e)
