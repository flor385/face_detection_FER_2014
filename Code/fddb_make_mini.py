"""
This script makes a subset of the FDDB database.
The purpose of the subset is to speed up development
and testing. The subset size can be adjusted.
"""

import logging
import sys
import os
import shutil

#   this points to the folder where the full FDDB is stored
#   we can't use fddb.PATH_ROOT because that might at any
#   time point to the subset
FDDB_FULL_DIR = "FDDB"

log = logging.getLogger(__name__)


def make_subset(subs_dir, subs_size):
    log.info("Creating subset, subs_dir: %s, size: %d",
             subs_dir, subs_size)

    #   create the subset folder
    if not os.path.exists(subs_dir):
        log.debug("Creating dir: %s", subs_dir)
        os.makedirs(subs_dir)

    #   copy subsets of fold textual descriptions
    for fold in range(1, 11):
        log.debug("Processing fold %d", fold)

        #   first handle file paths
        fold_paths = "FDDB-fold-{:02d}.txt".format(fold)
        with open(os.path.join(FDDB_FULL_DIR, fold_paths), "r") as f:
            img_paths = [l.strip() for l in f.readlines()[:subs_size]]
        #   and store them
        with open(os.path.join(subs_dir, fold_paths), "w") as f:
            f.write(os.linesep.join(img_paths))

        #   now copy the actual image files
        for img_path in img_paths:

            #   append the missing extension
            img_path += ".jpg"

            #   create folder if necessary
            full_path = os.path.join(subs_dir, img_path)
            if not os.path.exists(os.path.dirname(full_path)):
                os.makedirs(os.path.dirname(full_path))

            log.debug("Copying image to %s", full_path)
            shutil.copy(os.path.join(FDDB_FULL_DIR, img_path),
                        full_path)

        #   finally, copy the elipse info files
        fold_elipses = "FDDB-fold-{:02d}-ellipseList.txt".format(fold)
        with open(os.path.join(FDDB_FULL_DIR, fold_elipses), "r") as f:
            img_elipses = [l.strip() for l in f.readlines()]

        with open(os.path.join(subs_dir, fold_elipses), "w") as f:
            #   write lines until an image path that is not in img_paths
            for line in img_elipses:
                if (line.find("img") != -1) & (not line in img_paths):
                    break
                f.write(line)
                f.write(os.linesep)


def main():
    """
    Main program. Starts the subset creation.
    Command line arguments are:
        - path to FDDB subset location which defaults
          to "FDDB_subset"
        - subset size (number of photos per fold) which
          defaults to 15
    """
    logging.basicConfig(level=logging.DEBUG)
    log.info("FDDB subset creation main")

    #   extract parameters
    path = "FDDB_subset" if len(sys.argv) < 2 else sys.argv[1]
    subs_size = 15 if len(sys.argv) < 3 else int(sys.argv[2])

    #   run it
    make_subset(path, subs_size)


if __name__ == "__main__":
    main()
