from torch_geometric.transforms import BaseTransform

from data.fragmentations.fragmentations import FragmentRepresentation, Magnet

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
         'a1c75b2556c860b156e81e4eab3ccc9c',
         '4c97295c4d10fd438512879e76a3fa94',
         '8e29dc0ab3e5d80bb8a4899015ace6ba',
         'eac746408de2dfc79940e1d9fa12da5e',
         '47ed531c1fae1c5dfdcceedafa6a43d9',
         '41424c8fc83d7f65164d6f92257b995e',
         'b836934d75376e379f2f88c3dceb0f1d',
         '1b8ff794b97f932d3013b492488b5e63',
         'c7ad486489a14d91a0fc0fa2b5f2b179',
         '9dcccba557c08849c193295189cf7653',
         '6b114241cad0d9c4097c603df6ff8c10',
         '794ac3703ee6c9480d889a65e5127745',
         '833c89e8aca800e5707fb47ea63f9088',
         '161915877118e31c7ad6d7db48d50cd8',
         'b4b7f9310f775d80f7a3601b32a8845e',
         '16ca5746901379b93935c1fb14d49a8b',
         'd9b2de5cdc02d9b669e01b8e24fd7bfe',
         'ea64ed8710ae79e786b59eb22e86e1a8',
         '1b146c0365a13411191347bfcee6d38e',
         '9f7e1930e99a261f250404639aae6ec3',
         'bad967617266056b29d6eb2db59470b7',
         '63455e47e713ce9bee6e5a0f0a18bcdf',
         '8f9556ded5862d313741f5f580f26f44',
         'd1caf4b2885163918f42c5f9786061e2',
         'b9ba12113c92f2571d61f7a7f9b11a9d',
         '51b6be3aaac757475af72146b0f46f9f',
         '8bd60fe831e78beda7f820a0358c3136',
         '1a827d76aaa73ef0af9d2688d6bc8338',
         'e0669582681ae5929d51416018b30292',
         '1d2d649f6f9067a3f4e7909215fae014',
         '0d4975981166b2874d2e3fb8df49ba97',
         'ecca5668caefb06a5a6d1b6fbca28ad8',
         '31a3456b18d04b651166e80e093d2dd6',
         '082b9bcb2fa0fa1f0d74ef23b7b4833d',
         'af06040e912f05913d5fd697c5d560ab',
         '73f6531b6dca88eef48351cb2144b1ed',
         'b7d666c844c4325abcdebdc2954876eb',
         'f8b3abad4b03dec416dfe9226eb68fc8',
         '5184058f33558b291f152f6e5acb342e',
         'ae89ed3cd3a3daf867ac1e5c76cb7bf2',
         '63c42c8546f3d50ad7113fa3cb077ec5',
         'ba000c310bc33624a9da4d9b656b557c',
         '637737ef546c92f010d004e6d87be6d2',
         '37a09a4c353eea2d6ff35ef57141a278',
         '2705c06b68f4b5d8514e49b862debcd0',
         '824b534f22a184d32b92dca4e3eedee5',
         '3b6e4d7349ba46b6cbf4bf4b75296b94',
         '2b05bea826d4a0eff1cf4981626047fe',
         '52ceb8be15beb93495e9c78e23d37d60',
         '13693ea7ec318a138d99442bf4d4be28',
         '6e12fc2814c5e701656303ca8eaa988d',
         '0f1cb58e3ecdeb3d6e559f85e2eec2ce',
         'eae174f287df2aeaee7096c441368de4',
         'b426b8aff164c6823b185eb6b091738d',
         'ce508d7b81ceaf4db6152f790de711a8',
         'e589b3447af1b13516994048fd31b8e2',
         '7bae084a1c4d6879a44ea89f717c0511',
         'd88ab662acc413e76eddb6f29b91a093',
         '54a4f08e74072f3b858b20de96a0b1b5',
         '70bf3203e50f1621ff6ac0be820be7ae',
         'f0d38dcfc5388ababe0ae911c5972bac',
         '6883f46007e93d57941ed1706cee4387',
         '0a5cd747a3ce6fcd4bbb4548f4b23b3c',
         '252e111816cb216d608b459552877e2d',
         '35bfd426ffe8689f290130c31c813efd',
         '584fd584b3bbdf139e27b8d14a9ba3ff',
         'bcfd2adb17a8869d5597c7fd35e97bdf',
         'd8dec33c78d184d1e5fa6f641b1cfa58',
         'f72b0288e1838e3fba4b04bd63f7ea71',
         '2e7df1cfafc6fc61cf95d5cacee51fa9',
         'fecf1e9a8f06ca27a85f3cbbc8e44729']  # hashes of 100 most common substructures on ZINC-subset computed by Magnet


class CountSubstructures(BaseTransform):
    """
    Transformation that adds the counts of a Magnet substructure as target.
    Used for the substructure count experiment.

    Args:
        vocab (list): The vocabulary for magnet, defaults to a pre-computed list of the 100 most common substructures in ZINC-subset .
        substructure_idx (int, optional): The index of the substructure to count. If None, all substructure counts are added.
    """

    def __init__(self, vocab=VOCAB, substructure_idx=None):
        self.magnet = Magnet(vocab)
        self.fragment_representation = FragmentRepresentation(len(vocab))
        assert (substructure_idx is None or substructure_idx < len(vocab))
        self.substructure_idx = substructure_idx

    def __call__(self, data):
        data = self.magnet(data)
        data = self.fragment_representation(data)
        data.y = data.fragments.sum(dim=0)
        if self.substructure_idx is not None:
            data.y = data.y[self.substructure_idx]
        return data
