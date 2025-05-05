import torch
import random



# --- słownik form --------------------------------
FORMS = {}

# --------  Ω(2,2,2) : 35 orbit -------------------
def _register_1(id_, fn):
    FORMS[(1, id_)] = fn

_register_1(216, lambda x,y: (x*y,))
_register_1(217, lambda x,y: (x**2+y**2,))
_register_1(218, lambda x,y: (x**2+y,))
_register_1(219, lambda x,y: (x**2,))
_register_1(220, lambda x,y: (x,))
_register_1(221, lambda x,y: (0*x,))

# --------  Ω(2,2) : 21 orbit -------------------
def _register_2(id_, fn):
    FORMS[(2, id_)] = fn

_register_2(21, lambda x,y: (x**2+y,y**2+x))
_register_2(22, lambda x,y: (x**2-y**2+x,2*x*y-y))
_register_2(23, lambda x,y: (x**2+y,x*y))
_register_2(24, lambda x,y: (x**2+y,y**2))
_register_2(25, lambda x,y: (x**2,y**2))
_register_2(26, lambda x,y: (x**2-y**2,x*y))
_register_2(27, lambda x,y: (x**2-x,x*y))
_register_2(28, lambda x,y: (x**2,x*y))
_register_2(29, lambda x,y: (x*y,x+y))
_register_2(210, lambda x,y: (x**2+y**2,x))
_register_2(211, lambda x,y: (x,x*y))
_register_2(212, lambda x,y: (x**2,y))
_register_2(213, lambda x,y: (x**2+y,x))
_register_2(214, lambda x,y: (x**2,x))
_register_2(215, lambda x,y: (x,y))
_register_2(216, lambda x,y: (x*y,0*x))
_register_2(217, lambda x,y: (x**2+y**2,0*x))
_register_2(218, lambda x,y: (x**2+y,0*x))
_register_2(219, lambda x,y: (x**2,0*x))
_register_2(220, lambda x,y: (x,0*x))
_register_2(221, lambda x,y: (0*x,0*x))

# --------  Ω(2,2,2) : 35 orbit -------------------
def _register_3(id_, fn):
    FORMS[(3, id_)] = fn

_register_3(31, lambda x,y: (x**2 + y,   y**2 + x,      x*y))
_register_3(32, lambda x,y: (x**2 - y**2 + x, 2*x*y - y, -3*x**2 + y**2))
_register_3(33, lambda x,y: (x**2 + y,y**2 + x,x*y + 0.5*x + 0.5*y))
_register_3(34, lambda x,y: (x**2,y**2+x,x*y))
_register_3(35, lambda x,y: (x**2,y**2,x*y))
_register_3(36, lambda x,y: (x**2,y**2,x+y))
_register_3(37, lambda x,y: (x**2+y,y**2,x))
_register_3(38, lambda x,y: (x**2+y,y**2+x,0*x))
_register_3(39, lambda x,y: (x**2-y**2+x,2*x*y-y,0*x))
_register_3(310, lambda x,y: (x**2,x*y,y))
_register_3(311, lambda x,y: (x**2+y,x*y,x))
_register_3(312, lambda x,y: (x**2+y,x*y,0*x))
_register_3(313, lambda x,y: (x**2,y**2,y))
_register_3(314, lambda x,y: (x**2+y,y**2,0*x))
_register_3(315, lambda x,y: (x**2,y**2,0*x))
_register_3(316, lambda x,y: (x**2-y**2,x*y,0*x))
_register_3(317, lambda x,y: (x**2,x*y,x))
_register_3(318, lambda x,y: (x**2-x,x*y,0*x))
_register_3(319, lambda x,y: (x**2,x*y,0*x))
_register_3(320, lambda x,y: (x*y,x,y))
_register_3(321, lambda x,y: (x**2+y**2,x,y))
_register_3(322, lambda x,y: (x**2,x,y))
_register_3(323, lambda x,y: (x*y,x+y,0*x))
_register_3(324, lambda x,y: (x**2+y**2,x,0*x))
_register_3(325, lambda x,y: (x,x*y,0*x))
_register_3(326, lambda x,y: (x**2,y,0*x))
_register_3(327, lambda x,y: (x**2+y,x,0*x))
_register_3(328, lambda x,y: (x**2,x,0*x))
_register_3(329, lambda x,y: (x,y,0*x))
_register_3(330, lambda x,y: (x*y,0*x,0*x))
_register_3(331, lambda x,y: (x**2+y**2,0*x,0*x))
_register_3(332, lambda x,y: (x**2+y,0*x,0*x))
_register_3(333, lambda x,y: (x**2,0*x,0*x))
_register_3(334, lambda x,y: (x,0*x,0*x))
_register_3(335, lambda x,y: (0*x,0*x,0*x))

# --------  Ω(2,2,2,2) : 4 dodatkowe orbit ----------
def _register_4(id_, fn):
    FORMS[(4, id_)] = fn


_register_4(41, lambda x,y: (x**2 + y,   y**2 + x,      x*y,0*x))
_register_4(42, lambda x,y: (x**2 - y**2 + x, 2*x*y - y, -3*x**2 + y**2,0*x))
_register_4(43, lambda x,y: (x**2 + y,y**2 + x,x*y + 0.5*x + 0.5*y,0*x))
_register_4(44, lambda x,y: (x**2,y**2+x,x*y,0*x))
_register_4(45, lambda x,y: (x**2,y**2,x*y,0*x))
_register_4(46, lambda x,y: (x**2,y**2,x+y,0*x))
_register_4(47, lambda x,y: (x**2+y,y**2,x,0*x))
_register_4(48, lambda x,y: (x**2+y,y**2+x,0*x,0*x))
_register_4(49, lambda x,y: (x**2-y**2+x,2*x*y-y,0*x,0*x))
_register_4(410, lambda x,y: (x**2,x*y,y,0*x))
_register_4(411, lambda x,y: (x**2+y,x*y,x,0*x))
_register_4(412, lambda x,y: (x**2+y,x*y,0*x,0*x))
_register_4(413, lambda x,y: (x**2,y**2,y,0*x))
_register_4(414, lambda x,y: (x**2+y,y**2,0*x,0*x))
_register_4(415, lambda x,y: (x**2,y**2,0*x,0*x))
_register_4(416, lambda x,y: (x**2-y**2,x*y,0*x,0*x))
_register_4(417, lambda x,y: (x**2,x*y,x,0*x))
_register_4(418, lambda x,y: (x**2-x,x*y,0*x,0*x))
_register_4(419, lambda x,y: (x**2,x*y,0*x,0*x))
_register_4(420, lambda x,y: (x*y,x,y,0*x))
_register_4(421, lambda x,y: (x**2+y**2,x,y,0*x))
_register_4(422, lambda x,y: (x**2,x,y,0*x))
_register_4(423, lambda x,y: (x*y,x+y,0*x,0*x))
_register_4(424, lambda x,y: (x**2+y**2,x,0*x,0*x))
_register_4(425, lambda x,y: (x,x*y,0*x,0*x))
_register_4(426, lambda x,y: (x**2,y,0*x,0*x))
_register_4(427, lambda x,y: (x**2+y,x,0*x,0*x))
_register_4(428, lambda x,y: (x**2,x,0*x,0*x))
_register_4(429, lambda x,y: (x,y,0*x,0*x))
_register_4(430, lambda x,y: (x*y,0*x,0*x,0*x))
_register_4(431, lambda x,y: (x**2+y**2,0*x,0*x,0*x))
_register_4(432, lambda x,y: (x**2+y,0*x,0*x,0*x))
_register_4(433, lambda x,y: (x**2,0*x,0*x,0*x))
_register_4(434, lambda x,y: (x,0*x,0*x,0*x))
_register_4(435, lambda x,y: (0*x,0*x,0*x,0*x))
_register_4(436, lambda x,y: (x**2 + y, y**2,     x*y,  x))
_register_4(437, lambda x,y: (x**2,     y**2,     x*y,  x))
_register_4(438, lambda x,y: (x**2,     y**2,     x,    y))
_register_4(439,lambda x,y: (x**2 - y**2, x*y,   x,    y))
_register_4(440,lambda x,y: (x**2, x*y,   x,    y))


# --------  Ω(2,2,2,2,2) : 1 dodatkowe orbit ----------
def _register_5(id_, fn):
    FORMS[(5, id_)] = fn


_register_5(51, lambda x,y: (x**2 + y,   y**2 + x,      x*y,0*x,0*x))
_register_5(52, lambda x,y: (x**2 - y**2 + x, 2*x*y - y, -3*x**2 + y**2,0*x,0*x))
_register_5(53, lambda x,y: (x**2 + y,y**2 + x,x*y + 0.5*x + 0.5*y,0*x,0*x))
_register_5(54, lambda x,y: (x**2,y**2+x,x*y,0*x,0*x))
_register_5(55, lambda x,y: (x**2,y**2,x*y,0*x,0*x))
_register_5(56, lambda x,y: (x**2,y**2,x+y,0*x,0*x))
_register_5(57, lambda x,y: (x**2+y,y**2,x,0*x,0*x))
_register_5(58, lambda x,y: (x**2+y,y**2+x,0*x,0*x,0*x))
_register_5(59, lambda x,y: (x**2-y**2+x,2*x*y-y,0*x,0*x,0*x))
_register_5(510, lambda x,y: (x**2,x*y,y,0*x,0*x))
_register_5(511, lambda x,y: (x**2+y,x*y,x,0*x,0*x))
_register_5(512, lambda x,y: (x**2+y,x*y,0*x,0*x,0*x))
_register_5(513, lambda x,y: (x**2,y**2,y,0*x,0*x))
_register_5(514, lambda x,y: (x**2+y,y**2,0*x,0*x,0*x))
_register_5(515, lambda x,y: (x**2,y**2,0*x,0*x,0*x))
_register_5(516, lambda x,y: (x**2-y**2,x*y,0*x,0*x,0*x))
_register_5(517, lambda x,y: (x**2,x*y,x,0*x,0*x))
_register_5(518, lambda x,y: (x**2-x,x*y,0*x,0*x,0*x))
_register_5(519, lambda x,y: (x**2,x*y,0*x,0*x,0*x))
_register_5(520, lambda x,y: (x*y,x,y,0*x,0*x))
_register_5(521, lambda x,y: (x**2+y**2,x,y,0*x,0*x))
_register_5(522, lambda x,y: (x**2,x,y,0*x,0*x))
_register_5(523, lambda x,y: (x*y,x+y,0*x,0*x,0*x))
_register_5(524, lambda x,y: (x**2+y**2,x,0*x,0*x,0*x))
_register_5(525, lambda x,y: (x,x*y,0*x,0*x,0*x))
_register_5(526, lambda x,y: (x**2,y,0*x,0*x,0*x))
_register_5(527, lambda x,y: (x**2+y,x,0*x,0*x,0*x))
_register_5(528, lambda x,y: (x**2,x,0*x,0*x,0*x))
_register_5(529, lambda x,y: (x,y,0*x,0*x,0*x))
_register_5(530, lambda x,y: (x*y,0*x,0*x,0*x,0*x))
_register_5(531, lambda x,y: (x**2+y**2,0*x,0*x,0*x,0*x))
_register_5(532, lambda x,y: (x**2+y,0*x,0*x,0*x,0*x))
_register_5(533, lambda x,y: (x**2,0*x,0*x,0*x,0*x))
_register_5(534, lambda x,y: (x,0*x,0*x,0*x,0*x))
_register_5(535, lambda x,y: (0*x,0*x,0*x,0*x,0*x))
_register_5(536, lambda x,y: (x**2 + y, y**2,     x*y,  x,0*x))
_register_5(537, lambda x,y: (x**2,     y**2,     x*y,  x,0*x))
_register_5(538, lambda x,y: (x**2,     y**2,     x,    y,0*x))
_register_5(539,lambda x,y: (x**2 - y**2, x*y,   x,    y,0*x))
_register_5(540,lambda x,y: (x**2, x*y,   x,    y,0*x))
_register_5(541,lambda x,y: (x**2, x*y, y**2, x, y))

# ---------------------------------------------------
def get_form(n: int, id_: int):
    """Zwraca funkcję (x,y) -> tuple length n."""
    if n > 5:
    # id_==0 → wybierz "generic" (541);   w przeciwnym razie id_ odpowiada orbicie n=5
        id5 = id_ or 541
        base = FORMS[(5, id5)]
        def pad(x,y):
            return base(x,y) + (0*x,)*(n-5)
        return pad
    return FORMS[(n, id_)]




# wygodne aliasy: losowy id dla danego n


def random_form(n:int, rng=None):
    rng = rng or random
    if n > 5:
        # losuj spośród zdef. orbit n=5, potem pad‑zeros
        id5 = rng.choice([k[1] for k in FORMS if k[0]==5])
        return get_form(n, id5)
    choice = [k for k in FORMS if k[0]==n]
    return get_form(*rng.choice(choice))

