from torch_geometric.transforms import BaseTransform
from datasets.fragmentations.fragmentations import Magnet
from datasets.fragmentations.fragmentations import FragmentRepresentation

# hashes of 30 most common substructures on ZINC-subset computed by Magnet
VOCAB = ['da3c81488557dc57e6e1f0bd43d65336',
 -1,
 '5144181ac27497fdfa9bdb5b8b799630',
 '3ba4ffe16dfe637510ed1c3676ec6cb0',
 '081a3ef2a02910794db9cb14b9d27e2c',
 '69144809aea48cb46eae9c3950f24a15',
 '5a8eac0760a558d4174437be478ec0aa',
 '5f3aeb4dec7acf673e4c4a925f94174f',
 'c502b67eb6d91d909ba398fa39bec60c',
 '50803b752054f0512687e537ee7368f9',
 'cde6b48ed870286595c1455af7aff8bd',
 'ac0efc2073bac47c38268354d6d51e58',
 'bd4d5e09460e489b3a7687d2fb06fc0c',
 'b4844a241e7c75ea7eb690acd3c4c004',
 '653d67ad3d369bf2cd63d0027cde92ec',
 'a09fd8263c85c42edd74a3289977a8b3',
 '44003dd512e49164eb3c5d08bf21eb36',
 '7efca02d91f49e4bd32e9ceac2c5c6eb',
 'b301caa8ee54d69b7ea37306c72194d5',
 'ed17940d27aebadda70c31c5b11d2e16',
 '20a60ed013bc1976376f734be7d8d92c',
 'f848d40cbfa43815a8aa73d15a4c0574',
 '8fbe427a157a7273f48a5e3202b8050f',
 'dc09eb5accc563a82d933e0dde59da6e',
 '8565545eb7f3a8e7288f9a7afe487873',
 'a453c461e440ada28f922b2e2409e47b',
 '4fb41e08b5337e93adda663354c5ea93',
 '9915ff93784d747e82bcd9a73a2399e5',
 '74a9039e61abdfc370e0e9ccbda58085',
 'a1c75b2556c860b156e81e4eab3ccc9c']

class CountSubstructures(BaseTransform):

    def __init__(self, vocab = VOCAB, substructure_idx = None):
        self.magnet = Magnet(vocab)
        self.fragment_representation = FragmentRepresentation(len(vocab))
        assert(substructure_idx is None or substructure_idx < len(vocab))
        self.substructure_idx = substructure_idx
    
    def __call__(self, data):
        data = self.magnet(data)
        data = self.fragment_representation(data)
        data.y = data.fragments.sum(dim = 0)
        if self.substructure_idx is not None:
            data.y = data.y[self.substructure_idx]
        return data