from pytest_check import check
from graph_instances import random_fcs_graph


def test_fcs_generation(rng):
    G = random_fcs_graph(200, community_size=50, mu=0.3, average_degree=5, rng=rng)

    check.equal(len(G.nodes), 200)

    # check that the edges are as expected

    for e in G.edges:
        if e not in expected_edges:
            assert False, f"Edge {e} was not found in graph instance"


expected_edges = [
    (1, 40),
    (1, 97),
    (1, 172),
    (1, 91),
    (1, 69),
    (1, 146),
    (1, 88),
    (1, 194),
    (1, 8),
    (1, 82),
    (1, 99),
    (1, 100),
    (1, 66),
    (1, 187),
    (1, 171),
    (1, 158),
    (1, 63),
    (1, 113),
    (1, 20),
    (1, 2),
    (1, 84),
    (1, 195),
    (1, 27),
    (1, 4),
    (1, 138),
    (1, 77),
    (1, 29),
    (1, 191),
    (1, 17),
    (1, 141),
    (1, 115),
    (1, 155),
    (1, 85),
    (1, 165),
    (1, 50),
    (1, 3),
    (1, 190),
    (1, 142),
    (1, 178),
    (1, 144),
    (1, 139),
    (1, 22),
    (1, 98),
    (1, 128),
    (1, 51),
    (1, 28),
    (1, 150),
    (1, 167),
    (1, 127),
    (1, 90),
    (1, 198),
    (1, 62),
    (1, 157),
    (1, 156),
    (1, 65),
    (1, 175),
    (1, 181),
    (1, 71),
    (1, 170),
    (1, 200),
    (1, 81),
    (1, 176),
    (1, 10),
    (1, 160),
    (1, 61),
    (1, 111),
    (1, 6),
    (1, 41),
    (1, 136),
    (1, 92),
    (1, 164),
    (1, 79),
    (1, 131),
    (1, 185),
    (1, 154),
    (1, 117),
    (1, 114),
    (1, 174),
    (1, 5),
    (1, 9),
    (1, 120),
    (1, 103),
    (1, 112),
    (1, 94),
    (1, 143),
    (1, 109),
    (1, 33),
    (1, 101),
    (1, 184),
    (1, 39),
    (1, 105),
    (1, 53),
    (1, 199),
    (1, 134),
    (1, 21),
    (1, 60),
    (1, 45),
    (1, 56),
    (1, 108),
    (1, 147),
    (1, 163),
    (1, 76),
    (1, 26),
    (1, 74),
    (1, 38),
    (1, 7),
    (1, 135),
    (1, 68),
    (1, 177),
    (1, 34),
    (1, 96),
    (1, 145),
    (1, 102),
    (1, 121),
    (1, 119),
    (1, 188),
    (1, 49),
    (1, 189),
    (1, 35),
    (1, 31),
    (1, 25),
    (1, 122),
    (1, 166),
    (1, 75),
    (1, 24),
    (1, 182),
    (1, 70),
    (1, 42),
    (1, 140),
    (1, 93),
    (1, 57),
    (1, 197),
    (1, 125),
    (1, 148),
    (1, 106),
    (1, 43),
    (1, 137),
    (1, 59),
    (1, 129),
    (1, 83),
    (1, 64),
    (1, 52),
    (1, 196),
    (1, 36),
    (1, 30),
    (1, 123),
    (1, 151),
    (1, 118),
    (1, 87),
    (1, 107),
    (1, 11),
    (1, 183),
    (1, 110),
    (1, 16),
    (1, 133),
    (1, 54),
    (1, 149),
    (1, 153),
    (1, 46),
    (1, 72),
    (1, 192),
    (1, 58),
    (1, 32),
    (1, 152),
    (1, 1),
    (1, 95),
    (1, 55),
    (1, 124),
    (1, 130),
    (1, 159),
    (1, 15),
    (1, 104),
    (1, 116),
    (1, 14),
    (1, 23),
    (1, 44),
    (1, 80),
    (2, 109),
    (2, 38),
    (2, 51),
    (2, 143),
    (2, 36),
    (2, 89),
    (2, 39),
    (2, 185),
    (2, 31),
    (2, 53),
    (2, 159),
    (2, 200),
    (2, 45),
    (2, 94),
    (2, 163),
    (2, 66),
    (2, 61),
    (2, 136),
    (2, 43),
    (2, 116),
    (2, 75),
    (2, 6),
    (2, 121),
    (2, 196),
    (2, 60),
    (2, 59),
    (2, 69),
    (2, 92),
    (2, 48),
    (2, 3),
    (2, 153),
    (2, 49),
    (2, 54),
    (2, 81),
    (2, 57),
    (2, 46),
    (2, 114),
    (2, 74),
    (2, 130),
    (2, 26),
    (2, 195),
    (2, 21),
    (2, 30),
    (2, 186),
    (2, 189),
    (2, 120),
    (2, 13),
    (2, 17),
    (2, 24),
    (2, 11),
    (2, 90),
    (2, 131),
    (2, 178),
    (2, 152),
    (2, 4),
    (2, 41),
    (2, 33),
    (2, 127),
    (2, 176),
    (2, 117),
    (2, 20),
    (2, 76),
    (2, 102),
    (2, 110),
    (2, 165),
    (2, 95),
    (2, 132),
    (2, 62),
    (2, 78),
    (2, 169),
    (2, 193),
    (2, 123),
    (2, 91),
    (2, 157),
    (2, 145),
    (2, 28),
    (2, 93),
    (2, 73),
    (2, 40),
    (2, 5),
    (2, 9),
    (2, 32),
    (2, 118),
    (2, 111),
    (2, 100),
    (2, 113),
    (2, 37),
    (2, 142),
    (2, 27),
    (2, 112),
    (2, 10),
    (2, 151),
    (2, 80),
    (2, 63),
    (2, 140),
    (2, 126),
    (2, 155),
    (2, 14),
    (2, 158),
    (2, 150),
    (2, 181),
    (2, 22),
    (2, 12),
    (2, 82),
    (2, 183),
    (2, 101),
    (2, 104),
    (2, 156),
    (2, 55),
    (2, 108),
    (2, 47),
    (2, 23),
    (2, 29),
    (2, 79),
    (2, 197),
    (2, 64),
    (2, 192),
    (2, 175),
    (2, 77),
    (2, 15),
    (2, 86),
    (2, 168),
    (2, 179),
    (2, 138),
    (2, 173),
    (2, 188),
    (2, 115),
    (2, 35),
    (2, 85),
    (2, 170),
    (2, 177),
    (2, 52),
    (2, 182),
    (2, 99),
    (2, 161),
    (2, 103),
    (2, 42),
    (2, 172),
    (2, 2),
    (2, 88),
    (2, 67),
    (2, 65),
    (2, 7),
    (2, 105),
    (2, 139),
    (2, 68),
    (2, 106),
    (2, 147),
    (2, 154),
    (2, 8),
    (2, 166),
    (2, 50),
    (2, 199),
    (2, 135),
    (2, 133),
    (2, 98),
    (2, 167),
    (2, 119),
    (2, 174),
    (2, 162),
    (2, 160),
    (2, 124),
    (2, 180),
    (3, 52),
    (3, 186),
    (3, 189),
    (3, 101),
    (3, 138),
    (3, 199),
    (3, 94),
    (3, 51),
    (3, 55),
    (3, 114),
    (3, 60),
    (3, 31),
    (3, 40),
    (3, 3),
    (3, 82),
    (3, 166),
    (3, 16),
    (3, 116),
    (3, 180),
    (3, 37),
    (3, 41),
    (3, 175),
    (3, 181),
    (3, 155),
    (3, 134),
    (3, 42),
    (3, 28),
    (3, 137),
    (3, 129),
    (3, 182),
    (3, 72),
    (3, 13),
    (3, 127),
    (3, 172),
    (3, 119),
    (3, 25),
    (3, 14),
    (3, 43),
    (3, 160),
    (3, 39),
    (3, 105),
    (3, 150),
    (3, 115),
    (3, 61),
    (3, 27),
    (3, 88),
    (3, 44),
    (3, 168),
    (3, 169),
    (3, 9),
    (3, 103),
    (3, 143),
    (3, 135),
    (3, 195),
    (3, 196),
    (3, 108),
    (3, 191),
    (3, 20),
    (3, 177),
    (3, 90),
    (3, 163),
    (3, 17),
    (3, 183),
    (3, 62),
    (3, 171),
    (3, 49),
    (3, 174),
    (3, 140),
    (3, 22),
    (3, 64),
    (3, 187),
    (3, 97),
    (3, 146),
    (3, 170),
    (3, 156),
    (3, 96),
    (3, 153),
    (3, 54),
    (3, 197),
    (3, 142),
    (3, 164),
    (3, 136),
    (3, 21),
    (3, 106),
    (3, 66),
    (3, 18),
    (3, 190),
    (3, 59),
    (3, 107),
    (3, 57),
    (3, 124),
    (3, 123),
    (3, 87),
    (3, 86),
    (3, 133),
    (3, 69),
    (3, 159),
    (3, 67),
    (3, 47),
    (3, 45),
    (3, 71),
    (3, 95),
    (3, 131),
    (3, 35),
    (3, 126),
    (3, 11),
    (3, 165),
    (3, 8),
    (3, 38),
    (3, 70),
    (3, 56),
    (3, 91),
    (3, 53),
    (3, 122),
    (3, 23),
    (3, 24),
    (3, 176),
    (3, 185),
    (3, 99),
    (3, 118),
    (3, 121),
    (3, 58),
    (3, 6),
    (3, 184),
    (3, 158),
    (3, 130),
    (3, 200),
    (3, 110),
    (3, 111),
    (3, 125),
    (3, 92),
    (3, 144),
    (3, 117),
    (3, 120),
    (3, 36),
    (3, 128),
    (3, 73),
    (3, 147),
    (3, 74),
    (3, 7),
    (3, 98),
    (3, 161),
    (3, 30),
    (3, 141),
    (3, 4),
    (3, 76),
    (3, 104),
    (3, 63),
    (3, 26),
    (3, 139),
    (3, 113),
    (3, 89),
    (3, 173),
    (3, 10),
    (3, 34),
    (3, 198),
    (3, 19),
    (3, 145),
    (3, 149),
    (3, 80),
    (3, 29),
    (3, 152),
    (3, 93),
    (3, 162),
    (3, 50),
    (3, 100),
    (3, 32),
    (3, 85),
    (3, 154),
    (3, 81),
    (3, 157),
    (4, 97),
    (4, 69),
    (4, 99),
    (4, 38),
    (5, 114),
    (5, 153),
    (5, 117),
    (5, 145),
    (6, 126),
    (6, 191),
    (6, 196),
    (6, 130),
    (7, 103),
    (7, 146),
    (7, 75),
    (7, 177),
    (7, 153),
    (8, 32),
    (8, 48),
    (8, 129),
    (8, 77),
    (8, 194),
    (9, 75),
    (9, 65),
    (9, 117),
    (9, 134),
    (9, 70),
    (9, 39),
    (9, 199),
    (10, 157),
    (10, 198),
    (11, 56),
    (11, 40),
    (11, 199),
    (11, 70),
    (11, 137),
    (11, 128),
    (11, 106),
    (12, 128),
    (12, 63),
    (12, 84),
    (12, 140),
    (12, 33),
    (12, 106),
    (13, 21),
    (13, 44),
    (13, 81),
    (13, 149),
    (14, 51),
    (14, 61),
    (14, 84),
    (14, 106),
    (14, 152),
    (14, 126),
    (14, 125),
    (14, 94),
    (14, 54),
    (15, 131),
    (15, 55),
    (15, 88),
    (15, 65),
    (16, 70),
    (16, 100),
    (16, 55),
    (16, 109),
    (16, 46),
    (16, 198),
    (16, 98),
    (17, 143),
    (17, 80),
    (18, 71),
    (18, 162),
    (18, 77),
    (18, 179),
    (18, 80),
    (18, 24),
    (18, 124),
    (19, 61),
    (19, 58),
    (19, 155),
    (19, 82),
    (20, 173),
    (20, 51),
    (20, 82),
    (20, 193),
    (21, 75),
    (21, 191),
    (22, 52),
    (22, 150),
    (22, 84),
    (22, 40),
    (22, 200),
    (22, 54),
    (23, 56),
    (23, 129),
    (23, 84),
    (23, 34),
    (24, 133),
    (24, 180),
    (24, 165),
    (24, 87),
    (24, 178),
    (24, 182),
    (25, 85),
    (25, 31),
    (26, 98),
    (26, 44),
    (26, 69),
    (26, 100),
    (26, 171),
    (27, 87),
    (27, 167),
    (27, 59),
    (27, 77),
    (27, 191),
    (27, 56),
    (28, 172),
    (28, 72),
    (28, 149),
    (28, 58),
    (28, 131),
    (29, 178),
    (29, 160),
    (29, 56),
    (30, 97),
    (30, 134),
    (30, 105),
    (31, 156),
    (31, 45),
    (31, 54),
    (31, 80),
    (32, 176),
    (32, 146),
    (32, 96),
    (32, 49),
    (33, 127),
    (34, 173),
    (34, 153),
    (34, 80),
    (35, 195),
    (35, 77),
    (35, 145),
    (35, 151),
    (36, 123),
    (36, 110),
    (36, 167),
    (36, 48),
    (37, 72),
    (38, 197),
    (38, 87),
    (38, 132),
    (39, 81),
    (39, 109),
    (40, 95),
    (40, 148),
    (40, 134),
    (40, 182),
    (40, 157),
    (40, 108),
    (40, 129),
    (40, 124),
    (41, 121),
    (41, 139),
    (41, 52),
    (43, 172),
    (43, 79),
    (43, 100),
    (43, 138),
    (43, 80),
    (44, 64),
    (44, 200),
    (44, 189),
    (45, 157),
    (45, 178),
    (45, 151),
    (46, 176),
    (46, 112),
    (46, 126),
    (47, 181),
    (47, 148),
    (47, 134),
    (47, 145),
    (47, 54),
    (47, 194),
    (47, 114),
    (47, 49),
    (48, 119),
    (48, 60),
    (48, 199),
    (49, 180),
    (49, 156),
    (50, 101),
    (50, 177),
    (50, 190),
    (51, 164),
    (51, 108),
    (51, 124),
    (51, 143),
    (51, 65),
    (51, 145),
    (51, 162),
    (52, 75),
    (52, 151),
    (53, 162),
    (53, 120),
    (53, 186),
    (54, 58),
    (54, 187),
    (54, 171),
    (55, 103),
    (56, 139),
    (56, 162),
    (56, 160),
    (56, 148),
    (56, 90),
    (57, 100),
    (57, 157),
    (58, 167),
    (58, 176),
    (59, 144),
    (59, 64),
    (59, 166),
    (60, 69),
    (60, 133),
    (61, 108),
    (61, 141),
    (62, 81),
    (62, 151),
    (62, 83),
    (62, 171),
    (62, 127),
    (63, 64),
    (63, 170),
    (63, 107),
    (64, 65),
    (64, 93),
    (65, 166),
    (65, 68),
    (65, 149),
    (66, 88),
    (67, 180),
    (67, 77),
    (67, 158),
    (67, 69),
    (68, 99),
    (68, 74),
    (69, 128),
    (69, 170),
    (69, 178),
    (69, 126),
    (69, 188),
    (70, 166),
    (71, 124),
    (71, 72),
    (71, 109),
    (72, 191),
    (72, 197),
    (73, 90),
    (73, 149),
    (73, 161),
    (73, 158),
    (73, 152),
    (73, 126),
    (73, 137),
    (73, 105),
    (73, 129),
    (73, 124),
    (73, 190),
    (73, 169),
    (74, 134),
    (74, 115),
    (74, 168),
    (74, 123),
    (74, 132),
    (74, 178),
    (74, 149),
    (74, 108),
    (75, 165),
    (75, 178),
    (75, 115),
    (76, 97),
    (76, 78),
    (77, 135),
    (77, 142),
    (77, 113),
    (77, 90),
    (78, 200),
    (78, 160),
    (79, 86),
    (79, 167),
    (79, 106),
    (80, 148),
    (80, 123),
    (80, 145),
    (80, 189),
    (80, 183),
    (82, 183),
    (82, 167),
    (82, 121),
    (82, 116),
    (82, 192),
    (83, 83),
    (83, 163),
    (83, 138),
    (83, 195),
    (84, 131),
    (84, 165),
    (84, 123),
    (85, 177),
    (86, 109),
    (86, 192),
    (86, 124),
    (87, 134),
    (88, 103),
    (88, 157),
    (88, 135),
    (88, 194),
    (88, 167),
    (88, 196),
    (88, 195),
    (89, 173),
    (89, 178),
    (90, 109),
    (90, 94),
    (90, 163),
    (91, 180),
    (91, 175),
    (91, 123),
    (91, 188),
    (92, 200),
    (93, 155),
    (93, 195),
    (93, 119),
    (94, 184),
    (94, 116),
    (94, 97),
    (94, 180),
    (94, 162),
    (95, 124),
    (96, 107),
    (97, 143),
    (97, 99),
    (98, 139),
    (98, 131),
    (98, 126),
    (98, 102),
    (98, 142),
    (99, 132),
    (99, 153),
    (99, 174),
    (99, 106),
    (100, 178),
    (100, 161),
    (100, 132),
    (100, 134),
    (100, 189),
    (101, 145),
    (101, 110),
    (101, 137),
    (102, 158),
    (102, 181),
    (103, 179),
    (103, 197),
    (103, 133),
    (103, 169),
    (104, 123),
    (104, 110),
    (104, 147),
    (105, 185),
    (105, 158),
    (106, 128),
    (106, 172),
    (106, 182),
    (106, 118),
    (107, 171),
    (108, 159),
    (108, 126),
    (108, 198),
    (108, 161),
    (109, 118),
    (109, 188),
    (110, 141),
    (110, 166),
    (110, 156),
    (111, 111),
    (111, 195),
    (112, 164),
    (112, 121),
    (112, 192),
    (112, 148),
    (113, 198),
    (113, 141),
    (113, 185),
    (114, 186),
    (114, 173),
    (115, 161),
    (115, 193),
    (115, 198),
    (115, 154),
    (116, 178),
    (116, 189),
    (116, 156),
    (116, 134),
    (118, 158),
    (118, 167),
    (119, 175),
    (120, 124),
    (120, 143),
    (121, 150),
    (121, 191),
    (121, 182),
    (121, 148),
    (122, 186),
    (122, 158),
    (123, 194),
    (123, 192),
    (124, 184),
    (124, 190),
    (125, 141),
    (125, 131),
    (126, 194),
    (126, 181),
    (127, 145),
    (128, 185),
    (128, 200),
    (130, 170),
    (132, 133),
    (132, 183),
    (133, 143),
    (133, 170),
    (133, 191),
    (134, 165),
    (134, 162),
    (134, 138),
    (135, 151),
    (135, 177),
    (136, 153),
    (137, 188),
    (139, 178),
    (140, 179),
    (140, 187),
    (140, 189),
    (140, 195),
    (142, 179),
    (143, 197),
    (143, 153),
    (146, 195),
    (146, 196),
    (147, 184),
    (147, 166),
    (147, 155),
    (147, 148),
    (148, 172),
    (149, 170),
    (150, 170),
    (151, 159),
    (151, 179),
    (151, 155),
    (152, 191),
    (153, 189),
    (155, 174),
    (156, 178),
    (157, 183),
    (157, 172),
    (157, 198),
    (157, 165),
    (158, 165),
    (159, 194),
    (159, 177),
    (159, 173),
    (160, 185),
    (160, 198),
    (161, 180),
    (163, 190),
    (165, 172),
    (166, 176),
    (168, 195),
    (168, 180),
    (170, 176),
    (171, 191),
    (178, 183),
    (179, 182),
    (180, 190),
    (182, 200),
    (182, 187),
    (183, 189),
    (190, 196),
    (195, 198),
    (195, 199),
]
