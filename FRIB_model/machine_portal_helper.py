from phantasy import MachinePortal, disable_warnings
from .utils import get_Dnum_from_pv, split_name_field_from_PV, suppress_outputs, _name_conversions


disable_warnings()
_mp = MachinePortal(machine='FRIB', segment='LINAC')


def get_MPelem_from_PVs(PVs: list, 
                        mp: MachinePortal = _mp) -> list or None:
    """
    Retrieves MachinePortal elements from a list of PVs.

    Args:
        PVs (list): List of PV strings.
        mp (MachinePortal): MachinePortal instance.

    Returns:
        list or None: List of MachinePortal elements corresponding to the PVs.
    """
    names = [split_name_field_from_PV(PV, return_device_name=False)[0] for PV in PVs]
    # Check if mp is provided, otherwise use the default MachinePortal instance
    if mp is None:
        mp = MachinePortal(machine='FRIB', segment='LINAC')
    mp_names = mp.get_all_names()
    mp_dnums = [get_Dnum_from_pv(mp_name) for mp_name in mp_names]
    elems = []
    for name in names:
        with suppress_outputs():
            elem = mp.get_elements(name=name)
        if len(elem) == 0:
            # Try replacements
            for orig, new in _name_conversions:
                with suppress_outputs():
                    elem = mp.get_elements(name=name.replace(orig, new))
                if len(elem) > 0:
                    break

            # If still not found, get elem from matching dnum
            if len(elem) == 0:
                i = mp_dnums.index(get_Dnum_from_pv(name))
                if i >= 0:
                    with suppress_outputs():
                        elem = mp.get_elements(name=mp_names[i])
        if len(elem) == 0:
            elems.append(None)
            print(f"MachinePortal element is not found for PV: {name}")
        else:
            elems.append(elem[0])
    return elems