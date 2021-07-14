
import argparse


def _get_parser():
    """
    Parse command line inputs for aroma.

    Returns
    -------
    parser.parse_args() : argparse dict
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plotting ets-matrix script "
        )
    )

    # Required options
    reqoptions = parser.add_argument_group("Required arguments")
    reqoptions.add_argument(
        "-d", "--directory",
        dest="dir",
        required=True,
        help="Input directory.",
        type=str,
        nargs='+'
    )
    reqoptions.add_argument(
        "-s", "--subject",
        dest="subject",
        required=True,
        help="Subject",
        type=str,
        nargs='+'
    )
    reqoptions.add_argument(
        "-nr", "--nroi",
        dest="nROI",
        required=True,
        help="numero de ROI",
        type=str,
        nargs='+'
    )
    return parser