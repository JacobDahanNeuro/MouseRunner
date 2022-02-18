# -*- coding: utf-8 -*-

# %% Import toolboxes
import os
import time
import random
import pickle
import pygame
import threading
import PySpinController
import numpy as np
from itertools import groupby
from datetime import datetime

# %% Default Parameters class
class DefaultParams:
    """
    This class generates and stores session-type specific parameters and
    randomizes trial and ISI structure for each session. Parameters and
    randomized trial timing (i.e., trial start times) are saved to the temporary
    working directory using day-by-day naming conventions.
    :param session_type: Type of session requested for generation by MouseRunner.
    :type session_type: str
    :param shock: Determines whether shocks are delivered on conditioning days (equivalent to NOT mouse.naive).
    :type shock: bool
    """
    def __init__(self,session_type,shock):
        self.params = self.habituation() if session_type == 'Habituation' else\
                      self.conditioning(shock) if session_type == 'Conditioning' else\
                      self.ofc() if session_type == 'OFC' else self.recall()
        print(self.params)
        self.trial_generator()
        self.isi_generator()
        self.trialize()

    def trial_generator(self):
        """
        This function generates pseudo-random trials of equal numbers of each
        trial type listed in class attribute PARAMS. Trial generation is repeated
        until no more than two back-to-back trials of a single trial-type are
        present.
        """
        trials_nonrandom = list(iter(self.params['cs_ids'])) * self.params['stim_num']
        while True:
            self.trials = random.sample(trials_nonrandom,len(trials_nonrandom))                    # random sample without replacement of all trials for randomized tone presentation
            max_consecutive = max([sum(1 for _ in group) for _, group in groupby(self.trials)])
            if max_consecutive > 2:
                continue
            else:
                self.iter_trials = iter(self.trials)
                return

    def isi_generator(self):
        """
        This function generates pseudo-random ISIs for each trial, drawing from
        a normal distribution around the average ISI listed in class attribute
        PARAMS with a standard deviation listed in the same attribute. ISI
        generation is repeated until no ISI less than one-half the average
        ISI is present.
        """
        while True:
            self.isis = np.random.normal(self.params['avg_isi'],
                                         self.params['std_isi'],
                                         len(self.trials))
            if any (isi < (self.params['avg_isi'] / 2) for isi in self.isis):
                continue
            else:
                return

    def trialize(self):
        """
        This function generates an array of trial start times from previously
        generated trial types and ISIs. By definition, the first trial start
        time is equivalent to the baseline period listed in class attribute
        PARAMS.
        """
        self.trial_duration    = self.isis + self.params['cs_duration']
        self.iter_duration     = iter(self.trial_duration)
        self.trial_start_times = np.array([self.params['baseline'],*self.trial_duration[:-1]])
        self.trial_start_times = list(self.trial_start_times.cumsum())

    def save(self,save_params,mice):
        """
        This function stores previously generated trial timing data and session
        parameters to the temporary working directory using day-by-day naming
        conventions.
        :param save_params: Packaged parameters for saving to appropriate directory using appropriate naming conventions.
        :type save_params: dict
        :param mice: Mouse name(s) formatted for saving conventions.
        :type mice: str
        """
        np.savetxt(os.path.join(save_params['tmp_path'],"day{}-{}-timing.csv".format(save_params['day'],mice)),
                               [self.trials,self.trial_start_times],
                                delimiter =", ",
                                fmt ='% s')
        with open(os.path.join(save_params['tmp_path'],"day{}-{}-params.pickle".format(save_params['day'],mice)),'wb') as fi:
                               pickle.dump(self.params,fi,protocol=pickle.HIGHEST_PROTOCOL)

    def habituation(self):
        """
        This function returns easily editable session parameters for
        HABITUATION sessions to fill out DEFAULT PARAMS class. All parameters
        are stored for future analyses. Shock ID must match a single CS ID or
        be none-type. Stimulus number refers to iterations of EACH CS ID.
        All times are in seconds.
        """
        return {'session_type'        : 'Habituation',
                'stim_num'            : 4,
                'baseline'            : 180,
                'avg_isi'             : 60,
                'std_isi'             : 15,
                'cs_duration'         : 30,
                'cs_ids'              : (0,1,2),
                'shock_duration'      : 1.0,
                'shock_id'            : None,
                'shock'               : False,
                'laser'               : False,
                'laser_addl_duration' : 10}

    def conditioning(self,shock):
        """
        This function returns easily editable session parameters for
        CONDITIONING sessions to fill out DEFAULT PARAMS class. All parameters
        are stored for future analyses. Shock ID must match a single CS ID or
        be none-type. Stimulus number refers to iterations of EACH CS ID.
        All times are in seconds.
        """
        return {'session_type'        : 'Conditioning',
                'stim_num'            : 5,
                'baseline'            : 180,
                'avg_isi'             : 60,
                'std_isi'             : 15,
                'cs_duration'         : 30,
                'cs_ids'              : (0,1),
                'shock_duration'      : 1.0,
                'shock_id'            : 1,
                'shock'               : True if shock else False,
                'laser'               : True,
                'laser_addl_duration' : 10}

    def ofc(self):
        """
        This function returns easily editable session parameters for
        OFC sessions to fill out DEFAULT PARAMS class. All parameters
        are stored for future analyses. Shock ID must match a single CS ID or
        be none-type. Stimulus number refers to iterations of EACH CS ID.
        All times are in seconds.
        """
        return {'session_type'        : 'OFC',
                'stim_num'            : 5,
                'baseline'            : 180,
                'avg_isi'             : 60,
                'std_isi'             : 15,
                'cs_duration'         : 30,
                'cs_ids'              : (0,2),
                'shock_duration'      : 1.0,
                'shock_id'            : 2,
                'shock'               : True,
                'laser'               : False,
                'laser_addl_duration' : 10}

    def recall(self):
        """
        This function returns easily editable session parameters for
        RECALL sessions to fill out DEFAULT PARAMS class. All parameters
        are stored for future analyses. Shock ID must match a single CS ID or
        be none-type. Stimulus number refers to iterations of EACH CS ID.
        All times are in seconds.
        """
        return {'session_type'        : 'Recall',
                'stim_num'            : 5,
                'baseline'            : 300,
                'avg_isi'             : 90,
                'std_isi'             : 15,
                'cs_duration'         : 30,
                'cs_ids'              : (0,1,2),
                'shock_duration'      : 1.0,
                'shock_id'            : None,
                'shock'               : False,
                'laser'               : False,
                'laser_addl_duration' : 10}

class MainApp:
    """
    This class initializes the session log, generates and saves session
    parameters and trial structure, manages audio, laser, shocker, and LED
    output, and oversees all behavioral operations.
    :param a: Previously initialized Arduino API.
    :type a: class nanpy.arduinoapi.ArduinoApi
    :param connection: Serial port connection to previously assigned device.
    :type connection: class nanpy.serialmanager.SerialManager
    :param session_params: Tone, save, and session-specific parameters packaged for ease of access.
    :type session_params: class __main__.SessionParams
    :param laserPin: Arduino pin designated for laser output trigger.
    :type laserPin: int
    :param shockerPin: Arduino pin designated for shocker output trigger.
    :type shockerPin: int
    :param ledPin: Arduino pin designated for LED output trigger.
    :type ledPin: int
    """
    def __init__(self,a,connection,session_params,laserPin,shockerPin,ledPin):
        self.a = a
        self.connection = connection
        self.params = session_params
        self.laserPin = laserPin
        self.shockerPin = shockerPin
        self.ledPin = ledPin
        self.mice = next(iter(self.params.save_params['mice'])) if len(self.params.save_params['mice']) == 1 else\
                    '+'.join(self.params.save_params['mice'])
        self.log = open(os.path.join(self.params.save_params['tmp_path'],"day{}-{}-log.txt".format(self.params.save_params['day'],self.mice)),"a")

    def fetch_defaults(self):
        """
        This function initializes and saves a DefaultParams object to generate
        all session parameters according to previously defined class attribute
        SESSION_PARAMS.
        """
        self.default_params = DefaultParams(self.params.session_type,self.params.shock)
        self.default_params.save(self.params.save_params,self.mice)
        print(self.default_params.trials)
        print(self.default_params.trial_start_times)

    def pause(self,pause):
        """
        This function is a prettier version of time.sleep with (currently)
        no additional functionality.
        """
        time.sleep(pause)

    def num2type(self,trial_type):
        """
        This function converts int-type trial assignment to string-type trial
        names for generation of trial-type specific tone generation appropriate
        to specific mouse tone assignments.
        """
        trial_type = 'cs1' if trial_type == 1 else \
                     'cs2' if trial_type == 2 else 'cs_minus'
        return trial_type

    def session(self):
        """
        This function oversees all session operations, controlling and logging
        audio, laser, shocker, and LED outputs according to data assigned to
        class attributes DEFAULT_PARAMS and PARAMS. Actual trial start times
        (t=(Cue Start - Pre-Laser Cue Duration)) are recorded and saved, along
        with shocker status and trial type.
        """
        actual_cs_start=[]
        cue=[]
        shock=[]
        for trial_num in range(len(self.default_params.trials)):
            trial_type     = next(self.default_params.iter_trials)
            trial_duration = next(self.default_params.iter_duration)
            self.print_lock.acquire()
            print("========================", file=self.log)
            print("Trial: {}, Type: {}".format(trial_num,trial_type), file=self.log)
            print("\r\b\b\b========================")
            print("Trial: {}, Type: {}".format(trial_num,trial_type))
            self.print_lock.release()
            self.trial_start = time.time()
            if self.default_params.params['laser']:
                threading.Thread(target=self.laserCtrl).start()
            self.pause(self.default_params.params['laser_addl_duration'])
            threading.Thread(target=self.ledCtrl).start()
            pygame.mixer.init()
            pygame.mixer.music.load(self.params.tone_params[self.num2type(trial_type)])
            pygame.mixer.music.play()
            actual_cs_start.append(time.time() - self.start_time)
            cue.append(trial_type)
            if trial_type == self.default_params.params['shock_id'] and self.default_params.params['shock']:
                threading.Timer((self.default_params.params['cs_duration'] - self.default_params.params['shock_duration']),\
                                  self.shockCtrl).start()
                shock.append(1)
            else:
                shock.append(0)
            self.pause(trial_duration - self.default_params.params['laser_addl_duration'])
            self.print_lock.acquire()
            print("========================", file=self.log)
            print("\r\b\b\b========================")
            self.print_lock.release()
        np.savetxt(os.path.join(self.params.save_params['tmp_path'],"day{}-{}-actualtiming.csv".format(self.params.save_params['day'],self.mice)),
                               [cue,shock,actual_cs_start],
                                delimiter =", ",
                                fmt ='% s')
        self.terminate.set()
        self.print_lock.acquire()
        print("End Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'), file=self.log)
        print("\bEnd Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.print_lock.release()
        pygame.mixer.quit()

    def run(self):
        """
        This function initializes all Spinnaker Camera devices in separate threads
        and holds all behavioral operations until receiving a change of state
        event from the camera threads signaling that the device parameters have
        been adjusted and that the cameras are ready for image acquisition.
        Following a baseline period previously defined in class attribute
        DEFAULT_PARAMS, behavior is initiated and completed, after which the
        logger is elegantly closed.
        """
        self.ready      = threading.Event()
        self.terminate  = threading.Event()
        self.print_lock = threading.Lock()
        self.cam_thread = threading.Thread(target=PySpinController.controller,args=(self.ready,self.terminate,self.print_lock,self.params))
        self.cam_thread.setDaemon(True)
        self.cam_thread.start()
        self.ready.wait()
        self.print_lock.acquire()
        print("Start Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'), file=self.log)
        print("\bStart Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.print_lock.release()
        self.start_time = time.time()
        self.pause(self.default_params.params['baseline'] - self.default_params.params['laser_addl_duration'])
        self.session()
        self.log.close()

    def laserCtrl(self):
        """
        This function controls the Arduino output pin designated for laser
        trigger control, turning the laser on and off according to parameters
        assigned in class attribute DEFAULT_PARAMS and logging all operations.
        """
        import time
        # self.print_lock.acquire()
        print("laserON", time.time() - self.trial_start, file=self.log)
        print("\blaserON", time.time() - self.trial_start)
        # self.print_lock.release()
        self.a.digitalWrite(self.laserPin, self.a.HIGH)
        self.pause(self.default_params.params['laser_addl_duration'])
        self.pause(self.default_params.params['cs_duration'])
        self.pause(self.default_params.params['laser_addl_duration'])
        self.a.digitalWrite(self.laserPin, self.a.LOW)
        # self.print_lock.acquire()
        print("laserOFF", time.time() - self.trial_start, file=self.log)
        print("\blaserOFF", time.time() - self.trial_start)
        # self.print_lock.release()

    def shockCtrl(self):
        """
        This function controls the Arduino output pin designated for shocker
        trigger control, turning the shocker on and off according to parameters
        assigned in class attribute DEFAULT_PARAMS and logging all operations.
        """
        import time
        # self.print_lock.acquire()
        print("shockON", time.time() - self.trial_start, file=self.log)
        print("\bshockON", time.time() - self.trial_start)
        # self.print_lock.release()
        self.a.digitalWrite(self.shockerPin, self.a.HIGH)
        self.pause(self.default_params.params['shock_duration'])
        self.a.digitalWrite(self.shockerPin, self.a.LOW)
        # self.print_lock.acquire()
        print("shockOFF", time.time() - self.trial_start, file=self.log)
        print("\bshockOFF", time.time() - self.trial_start)
        # self.print_lock.release()

    def ledCtrl(self):
        """
        This function controls the Arduino output pin designated for LED
        trigger control, turning the LED on and off according to parameters
        assigned in class attribute DEFAULT_PARAMS and logging all operations.
        """
        import time
        # self.print_lock.acquire()
        print("ledON", time.time() - self.trial_start, file=self.log)
        print("\bledON", time.time() - self.trial_start)
        # self.print_lock.release()
        self.a.digitalWrite(self.ledPin, self.a.HIGH)
        self.pause(self.default_params.params['cs_duration'])
        self.a.digitalWrite(self.ledPin, self.a.LOW)
        # self.print_lock.acquire()
        print("ledOFF", time.time() - self.trial_start, file=self.log)
        print("\bledOFF", time.time() - self.trial_start)
        # self.print_lock.release()


def main(a,connection,session_params,laserPin=4,shockerPin=7,ledPin=8):
    """
    This function is the highest level behavioral controller, initializing the
    behavior app MainApp, fetching default parameters, and starting the session.
    :param a: Previously initialized Arduino API.
    :type a: class nanpy.arduinoapi.ArduinoApi
    :param connection: Serial port connection to previously assigned device.
    :type connection: class nanpy.serialmanager.SerialManager
    :param session_params: Tone, save, and session-specific parameters packaged for ease of access.
    :type session_params: class __main__.SessionParams
    :param laserPin: Arduino pin designated for laser output trigger.
    :type laserPin: int
    :param shockerPin: Arduino pin designated for shocker output trigger.
    :type shockerPin: int
    :param ledPin: Arduino pin designated for LED output trigger.
    :type ledPin: int
    """
    main_app = MainApp(a,connection,session_params,laserPin,shockerPin,ledPin)
    main_app.fetch_defaults()
    main_app.run()