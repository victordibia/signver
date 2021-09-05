# Copyright 2020 Cloudera, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from signver.utils.data_utils import mkdir, load_json_file, save_json_file
import logging
import faiss
import os
import numpy as np
logger = logging.getLogger(__name__)


class FaissIndex():
    def __init__(self, dim, index_type="flatip", index_dir=None):
        """Methods to manage a FAISS index

        Args:
            dim (int): dimension of FAISS index
            index_type (str, optional): FAISS index type. Defaults to "flatip". Supported options are [flatip, flatl2]
            index_dir (str, optional): default directory to load index from. Defaults to None.
        """
        self.index_type = index_type
        self.id_map = {}

        if (index_dir):
            self.load(index_dir)
        else:
            if (index_type == "flatip"):
                # self.index = faiss.IndexFlatL2(dim)
                self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
            if (index_type == "flatl2"):
                self.index = faiss.IndexFlatL2(dim)

    def add(self, values, id_strings=None):
        """Add vectors to an existing index. Also updates id_map dictionary
         which maps index of each vector to a string of interest.

        Args:
            values (np.array): np.array with shape corresponding with the dimensions for the index (, d) where d is index dimension
            id_strings (list, optional): list of strings that are added to the id_map. Defaults to None.
        """
        if self.index_type == "flatl2":
            self.index.add(values)
        elif self.index_type == "flatip":
            id_keys = range(self.index.ntotal,
                            self.index.ntotal + len(id_strings))
            self.index.add_with_ids(values, np.array(id_keys))
            self.update_id_map(id_keys, id_strings)

    def update_id_map(self, id_keys, id_strings):
        """Internal method to update id_map.

        Args:
            id_keys ([int]): FAISS id
            id_strings ([str]): strings of interest
        """
        new_id_map = dict(zip(id_keys, id_strings))
        self.id_map = {**self.id_map, **new_id_map}

    def decode_ids(self, ids):
        """Recover strings of interest given a list of ids

        Args:
            ids ([str]): list of ids

        Returns:
            [str]: list of strings that correspond to ids
        """
        id_strings = [self.id_map[str(id)] for id in ids]
        return id_strings

    def search(self, query, k):
        """Query FAISS inde

        Args:
            query (np.array): np.array of query terms
            k (int): number of similar items to return per query term.

        Returns:
            tuple (np.array, np.array) : (np.array or similarity distances and FAISS ids)
        """
        distances, idx = self.index.search(query, k)
        return distances, idx

    def load(self, load_dir="faiss"):
        """Load FAISS index from disc

        Args:
            load_dir (str, optional): directory containing FAISS index. Defaults to "faiss".
        """
        self.index = faiss.read_index(os.path.join(load_dir, "faiss.index"))
        self.id_map = load_json_file(
            os.path.join(load_dir, "faiss.map"))

    def save(self, save_dir="faiss"):
        """Save FAISS index to disc

        Args:
            save_dir (str, optional): Directory to save FAISS index. Defaults to "faiss".
        """
        mkdir(save_dir)
        faiss.write_index(self.index, os.path.join(save_dir, "faiss.index"))
        save_json_file(os.path.join(save_dir,  "faiss.map"), self.id_map)
