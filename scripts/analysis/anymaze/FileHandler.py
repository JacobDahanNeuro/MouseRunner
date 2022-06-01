# -*- coding: utf-8 -*-

import os
import re
import GUIOperator
import numpy as np
import pandas as pd
from itertools import chain
from itertools import repeat
from itertools import starmap
from collections import deque
from operator import attrgetter
from operator import itemgetter
import HelperFunctions as Helper
import ClassDefinitions as Classes

def grab_files(GUI,load_archive,archive_only,data_ids,data_types,data_constants,overwrite_id,key,manual_files,default_dir):
    if GUI and not manual_files:
        FilePaths    = GUIOperator.run_gui(default_dir)
        Files        = Helper.file_load(FilePaths,load_archive,archive_only,data_ids,data_types)
        MatchedFiles = Helper.file_sort(Files,data_constants,overwrite_id,key)
    else:
        Files        = Helper.file_load(manual_files,load_archive,archive_only,data_ids,data_types)
        MatchedFiles = Helper.file_sort(Files,data_constants,overwrite_id,key)
    return MatchedFiles
