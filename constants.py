import pandas as pd

class constants:
    speedOfLight = 299792458 # m/s
    clight = 299792458 # m/s
    earthRadius = 6371e03 # m

    ifdict = {'G': ['1C','2W','5Q'],
              'E': ['1C','5Q','6C'],
              'R': ['1C','2C'],
              'C': ['2I','6I'],
              }

    fdict = {
        "G1": 1575.42e6,
        "G2": 1227.60e6,
        "G5": 1176.45e6,
        "G12IF": 2803.02e6,
        "G15IF": 2751.87e6,
        "G25IF": 2404.05e6,
        "E1": 1575.42e6,
        "E5": 1176.45e6,
        "E6": 1278.75e6,
        "E7": 1207.14e6,
        "E8": 1191.795e6,
        "E15IF": 2751.87e6,
        "E16IF": 2854.17e6,
        "E56IF": 2455.2e6,
        "C1": 1575.45e6,
        "C2": 1561.098e6, # 2I
        "C5": 1176.45e6,
        "C6": 1268.52e6,
        "C7": 1207.14e6,
        "C26IF": 2829.618e6,
        "C27IF": 2768.238e6,
        "R1": 1602e6,
        "R2": 1246e6,
        "R3": 1202.025e6,
        "R12IF": 2848e6,
        "J1": 1575.42e6,
        "J2": 1227.60e6,
        "J5": 1176.45e6,
        "J6": 1278.75e6,
        "I5": 1176.45e6,
        }
    
    gloslots = {"R01": 1,
                "R02": -4,
                "R03": 5,
                "R04": 6,
                "R05": 1,
                "R06": -4,
                "R07": 5,
                "R08": 6,
                "R09": -2,
                "R10": -7,
                "R11": 0,
                "R12": -1,
                "R13": -2,
                "R14": -7,
                "R15": 0,
                "R16": -1,
                "R17": 4,
                "R18": -3,
                "R19": 3,
                "R20": 2,
                "R21": 4,
                "R22": -3,
                "R23": 3,
                "R24": 2,
                }
    
    glofreq = {"R01": {"L1": gloslots["R01"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R01"]*437.5e3+fdict["R2"]},
               "R02": {"L1": gloslots["R02"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R02"]*437.5e3+fdict["R2"]},
               "R03": {"L1": gloslots["R03"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R03"]*437.5e3+fdict["R2"]},
               "R04": {"L1": gloslots["R04"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R04"]*437.5e3+fdict["R2"]},
               "R05": {"L1": gloslots["R05"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R05"]*437.5e3+fdict["R2"]},
               "R06": {"L1": gloslots["R06"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R06"]*437.5e3+fdict["R2"]},
               "R07": {"L1": gloslots["R07"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R07"]*437.5e3+fdict["R2"]},
               "R08": {"L1": gloslots["R08"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R08"]*437.5e3+fdict["R2"]},
               "R09": {"L1": gloslots["R09"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R09"]*437.5e3+fdict["R2"]},
               "R10": {"L1": gloslots["R10"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R10"]*437.5e3+fdict["R2"]},
               "R11": {"L1": gloslots["R11"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R11"]*437.5e3+fdict["R2"]},
               "R12": {"L1": gloslots["R12"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R12"]*437.5e3+fdict["R2"]},
               "R13": {"L1": gloslots["R13"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R13"]*437.5e3+fdict["R2"]},
               "R14": {"L1": gloslots["R14"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R14"]*437.5e3+fdict["R2"]},
               "R15": {"L1": gloslots["R15"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R15"]*437.5e3+fdict["R2"]},
               "R16": {"L1": gloslots["R16"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R16"]*437.5e3+fdict["R2"]},
               "R17": {"L1": gloslots["R17"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R17"]*437.5e3+fdict["R2"]},
               "R18": {"L1": gloslots["R18"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R18"]*437.5e3+fdict["R2"]},
               "R19": {"L1": gloslots["R19"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R19"]*437.5e3+fdict["R2"]},
               "R20": {"L1": gloslots["R20"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R20"]*437.5e3+fdict["R2"]},
               "R21": {"L1": gloslots["R21"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R21"]*437.5e3+fdict["R2"]},
               "R22": {"L1": gloslots["R22"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R22"]*437.5e3+fdict["R2"]},
               "R23": {"L1": gloslots["R23"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R23"]*437.5e3+fdict["R2"]},
               "R24": {"L1": gloslots["R24"]*562.5e3+fdict["R1"],
                       "L2": gloslots["R24"]*437.5e3+fdict["R2"]},
               }
    
    modeName = {1: 'RTK',
                2: 'Float',
                3: 'BIE',
                4: 'DGNSS',
                5: 'PPP',
                6: 'SCRAPPED',
                7: 'Stand-Alone',
                }

    standardSigs = ['C1C','C2I']

class Ellipsoids:
    class WGS84:
        flattening = 298.257223563 #
        eccentricitySquared = 2*flattening-flattening**2
        semiMajorAxis = 6378137.0 # m
        semiMinorAxis = 6356752.314245 # m

class Signals:
    class L1:
        frequency = 1575.42e6  # Hz
        wavelength = constants.speedOfLight / frequency  # m
        codeDataVariable = ['C1C', 'C1W', 'C1L']
        carrierDataVariable = ['L1C', 'L1W', 'L1L']
        dopplerDataVariable = ['D1C', 'D1W', 'D1L']
        cn0DataVariable = ['S1C', 'S1W', 'S1L']
        constellation = 'GPS'
        constellationShort = 'G'

    class L2:
        frequency = 1227.60e6  # Hz
        wavelength = constants.speedOfLight / frequency  # m
        codeDataVariable = ['C2L', 'C2W', 'C2S']
        carrierDataVariable = ['L2L', 'L2W', 'L2S']
        dopplerDataVariable = ['D2L', 'D2W', 'D2S']
        cn0DataVariable = ['S2L', 'S2W', 'S2S']
        constellation = 'GPS'
        constellationShort = 'G'

    class L5:
        frequency = 1176.45e06  # Hz
        wavelength = constants.speedOfLight / frequency  # m
        codeDataVariable = ['C5Q', 'C5I']
        carrierDataVariable = ['L5Q', 'L5I']
        dopplerDataVariable = ['D5Q', 'D5I']
        cn0DataVariable = ['S5Q', 'S5I']
        constellation = 'GPS'
        constellationShort = 'G'

    class E1:
        frequency = 1575.42e6  # Hz
        wavelength = constants.speedOfLight / frequency  # m
        codeDataVariable = ['C1C', 'C1X', 'C1Z']
        carrierDataVariable = ['L1C', 'L1X', 'L1Z']
        dopplerDataVariable = ['D1C', 'D1X', 'D1Z']
        cn0DataVariable = ['S1C', 'S1X', 'S1Z']
        constellation = 'Galileo'
        constellationShort = 'E'

    class E5a:
        frequency = 1176.45e6  # Hz
        wavelength = constants.speedOfLight / frequency  # m
        codeDataVariable = ['C5Q', 'C5X', 'C5Z']
        carrierDataVariable = ['L5Q', 'L5X', 'L5Z']
        dopplerDataVariable = ['D5Q', 'D5X', 'D5Z']
        cn0DataVariable = ['S5Q', 'S5X', 'S5Z']
        constellation = 'Galileo'
        constellationShort = 'E'

    class E5b:
        frequency = 1207.14e6  # Hz
        wavelength = constants.speedOfLight / frequency  # m
        codeDataVariable = ['C7Q', 'C7X', 'C7Z']
        carrierDataVariable = ['L7Q', 'L7X', 'L7Z']
        dopplerDataVariable = ['D7Q', 'D7X', 'D7Z']
        cn0DataVariable = ['S7Q', 'S7X', 'S7Z']
        constellation = 'Galileo'
        constellationShort = 'E'

    class E5:
        frequency = 1191.795e6  # Hz
        wavelength = constants.speedOfLight / frequency  # m
        codeDataVariable = ['C7Q', 'C7X', 'C7Z']
        carrierDataVariable = ['L7Q', 'L7X', 'L7Z']
        dopplerDataVariable = ['D7Q', 'D7X', 'D7Z']
        cn0DataVariable = ['S7Q', 'S7X', 'S7Z']
        constellation = 'Galileo'
        constellationShort = 'E'

    class E6:
        frequency = 1278.75e06  # Hz
        wavelength = constants.speedOfLight / frequency  # m
        codeDataVariable = ['C6C', 'C6X', 'C6Z']
        carrierDataVariable = ['L6C', 'L6X', 'L6Z']
        dopplerDataVariable = ['D6C', 'D6X', 'D6Z']
        cn0DataVariable = ['S6C', 'S6X', 'S6Z']
        constellation = 'Galileo'
        constellationShort = 'E'
