import pickle


# class MockConnection:
#     def __init__(self, path="."):
#         self.path = path
#         self.inverted_index = {}
#         self.data = {}
#         self.flushed = 0
#
#     def save(self, id, footprint, frame_index):
#         hash = self.hash_footprint(footprint)
#         if hash in self.inverted_index:
#             self.inverted_index[hash] += [(id, frame_index)]
#         else:
#             self.inverted_index[hash] = [(id, frame_index)]
#
#         if id in self.data:
#             self.data[id] += [footprint]
#         else:
#             self.data[id] = [footprint]
#
#     def hash_footprint(self, footprint):
#         return sum(footprint)
#
#     def query_by_id(self, id):
#         return self.data.get(id)
#
#     def query_by_footprint(self, fp):
#         return self.inverted_index.get(self.hash_footprint(fp), [])
#
#     def flush(self, db_name, path=None):
#         self.flushed += 1
#
#     @staticmethod
#     def merge_dicts(dict1, dict2):
#         d = {}
#         d.update(dict1)
#         for k, v in dict2.items():
#             if k in d:
#                 d[k] += v
#             else:
#                 d[k] = v
#         return d
#
#     def __add__(self, other):
#         new_conn = MockConnection()
#         new_conn.data = self.merge_dicts(self.data, other.data)
#         new_conn.inverted_index = self.merge_dicts(self.inverted_index, other.inverted_index)
#         return new_conn
