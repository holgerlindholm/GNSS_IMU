# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:06:10 2022

# ----------------------------------------------------------------------- #
#   RinexReader: Class for reading observations from Rinex 3 files one    #
#                one epoch at a time, or a whole file at once             #
#                                                                         #   
# ----------------------------------------------------------------------- #
#   Class     : RinexReader                                               #
#   Date      : 21-MAY-2021 EDITED for Pandas Oct. 17, 2022               #
                Reworked SEP-2024
                Rinex 2 reading added February 2025
#   Author    : Søren Reime Larsen                                        #
#               Geodesy and Earth Observation Group, DTU Space            #
#               Technical University of Denmark                           #
# ----------------------------------------------------------------------- #

TODO: Add a file incomplete check

@author: sorla
"""

import datetime
import numpy as np
import pandas as pd
import sys
import time as tm
import constants as cnst

class rinexReader:
    
    def __init__(self, path=False):
        
        if path:
            self.path = path
            if not isinstance(self.path, list):
                self.path = [self.path]
        
        self.period = [1, 0] # Update frequency
        self.obs = {}
        self.obsSvid = {}
        self.systems = []
        self.upreftime = 0
        self.out = pd.DataFrame()
        self.f = False
        
    def addFiles(self, path):
        """
        Add rinex paths to the rinexReader

        Parameters
        ----------
        path : str or list
            Path or Paths to add to the rinexReader - should be paths to rinex 
            files. Paths are added to self.path

        Returns
        -------
        None.

        """
        
        if hasattr(self, 'path'):
            if not isinstance(self.path, list):
                path=[path]
            self.path.extend(path)
        else:
            self.path = path
            if not isinstance(self.path, list):
                self.path = [self.path]
        
    def checkEpoch(self, epoch) -> bool:
        """
        Check if current epoch is within the wanted interval
        """
        if (self.startTime <= epoch <= self.endTime):
            # If epochs are missing print a warning
            if self.oldEpoch:
                delta = (epoch-self.oldEpoch).total_seconds()
                if delta > self.period[0]:
                    print(f'Gap in RINEX data from {self.fileName} @ {epoch.strftime("%H:%M:%S")}', flush=True)
            return True
        else:
            return False

    def openRnxFile(self, path):
    
        """
        Open RINEX 3 file and initialize reading
        """
        
        if hasattr(self, 'path'):
        
            try:
                # self.f = open(path)
                f = open(path)
                # print("Reading:", path)
            except:
                print("Path not found:", path)
                sys.exit()
            
            # print(f'Opening {path}', flush=True, end="")
                
            self.fileName = path.split("/")[-1]
            # self.fileName = self.fileName.split(".")[0]
            
            line = f.readline()
            if 'RINEX VERSION / TYPE' in line:
                self.fullversion = float(line[:10])
                self.version = int(line[:10].strip()[0])
                # print(f"RINEX version {self.version}")
            else:
                print("File error: No RINEX version detected in line 1")
                sys.exit()
            
            if self.version == 3:
                self.readRnx3Header(f)
            elif self.version == 2:
                print("Only RINEX3 files are supported")
                # self.readRnx2Header(f)
            else:
                print("Error: Wrong version. Only RINEX version 3 is supported.")
                sys.exit()
            
            return f
        else:
            print("No filepaths added...")
            return False
    
    def readFile(self,
                readConst=["G"], 
                sigTypes = ["C1C","L1C","S1C","D1C"],
                startTime=False,
                endTime=False,
                ):
        """
        Generic call to read file - can handle RINEX version (soon)
        Remember to use 1,2,3,5,6,7 for freqs depending on what data you want
    
        """
        
        if readConst:
            if not isinstance(readConst, list):
                readConst = [readConst]
        self.readConst = readConst
        
        if sigTypes == [""]:
            self.comb=False
        else:
            self.comb=sigTypes
    
        self.f = self.openRnxFile(self.path[0])
        
        self.oldEpoch = False
        
        if startTime:            
            self.startTime = startTime
        else:
            self.startTime = self.fileStart
            
        if endTime:
            self.endTime = endTime
        else:
            self.endTime = self.fileEnd
        
        self.timelist = []
        
        if self.f:
            if self.version == 3:
                self.readRnx3File(self.f)
            elif self.version == 2:
                print("Only RINEX3 files are supported")
                self.readRnx2File(self.f)
            else:
                print("Only RINEX version 3 is supported!")
                
            try:
                self.f.close()
            except:
                pass
        
    # def readData(self,
    #              startTime=False,
    #              endTime=False,
    #              ):

    #     self.oldEpoch = False
        
    #     if startTime:            
    #         self.startTime = startTime
    #     else:
    #         self.startTime = self.fileStart
            
    #     if endTime:
    #         self.endTime = endTime
    #     else:
    #         self.endTime = self.fileEnd
        
    #     self.timelist = []
        
    #     if self.f:
    #         if self.version == 3:
    #             self.readRnx3File(self.f)
    #         elif self.version == 2:
    #             print("Only RINEX3 files are supported")
    #             self.readRnx2File(self.f)
    #         else:
    #             print("Only RINEX version 3 is supported!")
                
    #         try:
    #             self.f.close()
    #         except:
    #             pass


#%% RINEX 3

    def readRnx3Header(self, f):
        """
        Read header, and store data internally
        Only observations types are read thus far

        """
        
        self.obsTypes = {}
        self.obsIdx = {}
        self.obsUse = {}
        self.obsMult = {}
        
        for line in f:
            if "END OF HEADER" in line:
                break
            
            elif 'SYS / # / OBS TYPES' in line[60:]:
                const = line[0]
                nObs = int(line[3:6])
                self.systems.append(const)
                self.obsTypes[const] = line[6:60].split()
                    
                n = nObs-13
                while n > 0:
                    # line = self.f.readline()
                    line = f.readline()
                    self.obsTypes[const] += line[6:60].split()
                    n -= 13
                
                if const in self.readConst:
                    if self.comb:
                        self.obsUse[const] = []
                        self.obsIdx[const] = []
                        self.obsMult[const] = []
                        for idx, obscode in enumerate(self.obsTypes[const]):
                            # if obscode[:2] in self.comb:
                            if obscode in self.comb:
                                self.obsIdx[const].append(idx)
                                self.obsUse[const].append(obscode)
                                if obscode[0] in ["L"]:
                                    if const =='R':
                                        if obscode[1]=='3':
                                            freq = cnst.constants.fdict[const+obscode[1]]
                                            self.obsMult[const].append(cnst.constants.clight/freq)
                                        else:
                                            self.obsMult[const].append(1)
                                    else:
                                        freq = cnst.constants.fdict[const+obscode[1]]
                                        self.obsMult[const].append(cnst.constants.clight/freq)
                                else:
                                    self.obsMult[const].append(1)
                        self.obsMult[const] = np.array(self.obsMult[const])
                    else:
                        self.obsUse[const] = self.obsTypes[const]
                        self.obsIdx[const] = [x for x in range(len(self.obsTypes[const]))]
                        for obscode in self.obsUse[const]:
                            if obscode[0] in ["L"]:
                                # HANDLE GLONASS FDMA SEPERATELY
                                if const =='R':
                                    if obscode[1]=='3':
                                        freq = cnst.constants.fdict[const+obscode[1]]
                                        self.obsMult[const].append(cnst.constants.clight/freq)
                                    else:
                                        self.obsMult[const].append(1)
                                else:
                                    freq = cnst.constants.fdict[const+obscode[1]]
                                    self.obsMult[const].append(cnst.constants.clight/freq)
                            else:
                                self.obsMult[const].append(1)
                        self.obsMult[const] = np.array(self.obsMult[const])
                
                assert len(self.obsTypes[const]) == nObs
            
            elif 'TIME OF FIRST OBS' in line[60:]:
                inp = line[:60].split()
                self.fileStart = datetime.datetime(int(inp[0]), int(inp[1]), int(inp[2]), int(inp[3]), int(inp[4]), int(inp[5].split(".")[0]))
            
            elif 'TIME OF LAST OBS' in line[60:]:
                inp = line[:60].split()
                self.fileEnd = datetime.datetime(int(inp[0]), int(inp[1]), int(inp[2]), int(inp[3]), int(inp[4]), int(inp[5].split(".")[0]))
            
            elif 'INTERVAL' in line[60:]:
                self.period = [int(x) for x in line[:60].strip().split(".")]
            
            elif 'APPROX POSITION XYZ' in line[60:]:
                self.approxPos = np.array([float(x) for x in line[:60].split()])
                
            elif 'ANTENNA: DELTA H/E/N' in line[60:]:
                self.deltaHen = np.array([float(x) for x in line[:60].split()])
            
            elif 'ANT # / TYPE' in line[60:]:
                antType = [x for x in line[:60].split()]
                self.antSerial = antType[0]
                self.antType = antType[1]
                try:
                    self.antRadome = antType[2]
                except:
                    pass
    
    def readRnx3File(self, f):
        """
        Read Rinex 3 Observation file, and store observations for all 
        constellations in "systems"
        
        Returns nothing - all data stored in self.obs

        """
        # print(f'Reading RINEX{self.version} observations from: {self.fileName}', flush=True)
        print(f'Reading RINEX{self.version} observations from {self.fileName}')
        self.readError = False

        # Read data line for line
        for line in f:
            if line[0] == '>':
                # Read the epoch
                h = line[2:].split()
                t = [int(float(n)) for n in h[0:6]]
                msec = int(float(h[5].split('.')[1])/10000)
                
                epoch = datetime.datetime(t[0], t[1], t[2], t[3], t[4], t[5])
                nSat = int(h[7])
                
                # Check if there are missing epochs
                if self.checkEpoch(epoch):
                    self.timelist.append(epoch)
                    # Read all observations
                    for d in range(nSat):
                        try:
                            line = f.readline()
                            const = line[0]
                            
                            self.readError = False
                        
                            if const in self.readConst:
                            # if const in self.obs:
                                svid = line[:3]
                                
                                if svid not in self.obsSvid:
                                    self.obsSvid[svid]={}
                                self.obsSvid[svid][epoch]={}
    
                                if epoch not in self.obs:
                                    self.obs[epoch]={}
                                    
                                # Get data from line    
                                data=[]
                                ll = len(self.obsTypes[const])*16+3
                                # Read all data
                                for idx, i in enumerate(range(3,ll,16)):
                                    if idx in self.obsIdx[const]:
                                        try:
                                            data.extend([round(float(line[i:i+16].split()[0]),3)])
                                        except:
                                            data.extend([np.nan])
    
                                # Store data in dict
                                # self.obs[epoch][svid] = {z:y for z,y in zip(self.obsUse[const], self.obsMult[const]*np.array(data))}
                                # self.obsSvid[svid][epoch] = {z:y for z,y in zip(self.obsUse[const], self.obsMult[const]*np.array(data))}
                                self.obs[epoch][svid] = {z:y for z,y in zip(self.obsUse[const], np.array(data))}
                                self.obsSvid[svid][epoch] = {z:y for z,y in zip(self.obsUse[const], np.array(data))}
                                if const == 'R':
                                    for sig in self.obs[epoch][svid]:
                                        if sig[:2] in ["L1", "L2"]:
                                            
                                            self.obs[epoch][svid][sig] *= cnst.constants.clight/cnst.constants.glofreq[svid]["L"+sig[1]]
                                            self.obsSvid[svid][epoch][sig] *= cnst.constants.clight/cnst.constants.glofreq[svid]["L"+sig[1]]
                        except:
                            if not self.readError:
                                self.readError = True
                                print(f"RINEX read error: {self.fileName} @ {epoch} {svid}")

                    # Store old epoch
                    self.oldEpoch = epoch
                else:
                    continue
        # Close RINEX file
        f.close()        
    
    def get_epoch_data(self, reftime, consts=['G'], oTypes=['C1C'], svidx=[], upsampling=False):
          
        if self.startTime <= reftime <= self.endTime:
            obs = pd.DataFrame.from_dict(self.obs[reftime], orient='index')
            obs = obs.iloc[[num for num, idx in enumerate(obs.index) if idx[0] in consts], :]
            obs = obs.iloc[:, [num for num, col in enumerate(obs.columns) if col in oTypes]]
            
            if any(svidx):
                obs = obs.iloc[[num for num, idx in enumerate(obs.index) if idx in svidx], :]
            
            testnans = np.sum(np.isnan(obs.values), axis=0)==len(obs)
            if np.any(testnans):
                obs = obs.loc[:, ~(testnans)]
            obs.sort_index(axis=1, inplace=True) # Sort by signal type
            obs.sort_index(inplace=True) # Sort by constellation
        
        # print(obs)
        else:
            obs = pd.DataFrame()
            print(f"Given time is not within RINEX file from {self.startTime} to {self.endTime}")
        
        return obs
    
    def get_obs_data(self, oTypes='C1C', svidx=[], upsampling=False):
        
        obsDict = {}
        
        if np.any(svidx):
            for sv in self.obsSvid:
                if sv in svidx:
                    try:
                        obs = pd.DataFrame.from_dict(self.obsSvid[sv], orient='index')
                        obsDict[sv] = obs.loc[:,oTypes]
                    except:
                        pass
        else:
            for sv in self.obsSvid:
                try:
                    obs = pd.DataFrame.from_dict(self.obsSvid[sv], orient='index')
                    obsDict[sv] = obs.loc[:,oTypes]
                except:
                    pass
            
        obs = pd.DataFrame.from_dict(obsDict, orient='index').T
        obs.sort_index(axis=1, inplace=True)
        
        return obs
    
    def get_svid_data(self, svid, oTypes=[], upsampling=False):
          
        if svid in self.obsSvid:
            obs = pd.DataFrame.from_dict(self.obsSvid[svid], orient='index')
            
            if any(oTypes):
                obs = obs.iloc[:, [num for num, col in enumerate(obs.columns) if col in oTypes]]
            
            testnans = np.sum(np.isnan(obs.values), axis=0)==len(obs)
            if np.any(testnans):
                obs = obs.loc[:, ~(testnans)]
            obs.sort_index(axis=1, inplace=True) # Sort by signal type
            obs.sort_index(inplace=True) # Sort by constellation
        
        # print(obs)
        else:
            obs = pd.DataFrame()
            print(f"Given satellite was not found in RINEX data")
        
        return obs