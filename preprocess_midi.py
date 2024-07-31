import mido
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import shutil
import pretty_midi
from collections import Counter
from mido import MidiFile, MidiTrack, Message



#####################################################################################
               #  Get melodies and select tempo + remove corrupted files
#####################################################################################

def search_melodies(path):
    """
    Searches for songs that have a 'MELODY' (monophonic) track, and non-zero total time

    Parameters:
    path: path locating the MIDI file
 
    Returns:
    has_melody: 1 if the track has melody and non-zero duration, 0 otherwise
    """
    has_melody = 0
    
    try:
        midi_data = pretty_midi.PrettyMIDI(path)
    except (OSError, IOError, EOFError) as e:
            #print(f"Warning: I/O error '{e}' encountered in {midi_file}. Skipping file.")
        return None  # Return None or handle this case as needed
    except Exception as e:
            #print(f"Warning: Error '{e}' encountered in {midi_file}. Skipping file.")
        return None
    
        
    midi_file = mido.MidiFile(path)
    
    # Check if track of Metadata exists
    track_names = [track.name for track in midi_file.tracks]
    if '' not in track_names:
        return None
        
    # Check if Melody track exists
    if 'MELODY'in track_names:
        
        for track in midi_file.tracks:
            
            #Check that it actually has notes
            if track.name=='MELODY':
                t = []
                for msg in track:
                    t.append(msg.time)
                if np.any(t!=0): 
                    has_melody = 1
                break
        
    return has_melody
    
    
    

def select_tempo(paths, tpb, tempo_num, tempo_den, metadata):
    """
    Selects track with specific tempo and ticks per beat
    
    Parameters:
    tpb (int): desired ticks per beat
    tempo_num (int): Desired numerator for the tempo
    tempo_den (int): Desired denominator for the tempo
    metadata (list): [n_songs x 3] list, for each song: ticks per beat, metamessage, current time

    Returns:
    indexes_ (list): indexes of paths corresponding to tracks satisfying the criteria
    selected_paths (list) : paths of tracks satisfying the criteria
    selected_meta (list) : metadata of selected midi files
    """
    # select 384 ticks per beat and 4/4 tempo
    indexes_ = []
    selected_meta = []
    selected_paths = []
    for i in range(len(metadata)):
        if metadata[i][0] == tpb:
            for msg in metadata[i][1]:
                if msg.type == 'time_signature' and msg.numerator == tempo_num and msg.denominator == tempo_den:
                    indexes_.append(i)
                    selected_paths.append(paths[i])
                    selected_meta.append(metadata[i])
                    break  # Exit the inner loop once a matching time_signature is found
    return(indexes_, selected_paths, selected_meta)
  
  
  
    
    
# Function to read MIDI file and extract note events in ticks
def extract_note_events(path, track_name):
    """
    Extract note events from tracks
    
    Parameters:
    path (str): path to the collection of midi files
    track_name (str): name of the track to select ('MELODY')

    Returns:
    events (list): sequence of notes (0-128) played 
    current_time (list) : cumulative time (in ticks) at which each note is played 
  
    """
    midi_file = mido.MidiFile(path)
    events = []
    current_time = 0

    for i, track in enumerate(midi_file.tracks):
        if track.name==track_name:
            for msg in track:
                current_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    events.append([int(current_time), int(msg.note), 1])   #1: note on
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    events.append([int(current_time), int(msg.note), 0])  #0: note off
    return events,current_time



def unpack_notes(notes, max_time):
    """
    Unpacks note events into a continuous time series representation (one note per tick)

    Parameters:
    notes (ndarray): Array of note events with columns [time, note, event_type]
                     where event_type is 1 for 'note on' and 0 for 'note off'.
    max_time (int): The length of the output time series.

    Returns:
    ndarray: A time series array representing the notes.
    """
    notes_unpacked = np.zeros((max_time,))

    # IIf there is silence at the start, the melody is considered to start after silence
    
    silence_time = notes[0, 0]
    
    # Process each note event
    for i in range(notes.shape[0] - 1):
        start_time = int(notes[i, 0]-silence_time)
        end_time = int(notes[i + 1, 0]-silence_time)
        note = notes[i, 1]
        event_type = notes[i, 2]

        if event_type == 1:
            # Note on: fill every time tick with that note
            notes_unpacked[start_time:end_time] = note
        else:
            # Note off: silence until the next note, filled with previous note
            notes_unpacked[start_time:end_time] = notes_unpacked[start_time-1]

    # Handle the last note event, extending to the end of the time series
    last_time = int(notes[-1, 0])
    if notes[-1, 2] == 1:
        notes_unpacked[last_time:] = notes[-1, 1]
    else:
        notes_unpacked[last_time:] = notes_unpacked[last_time-1]
        
    return notes_unpacked



def most_frequent(arr):
    """
    Select most frequent note in the time-step (note_length ticks) considered
    
    Parameters:
    arr (ndarray): note_length-dimensional array of notes (0-128)

    Returns:
    most_common[0][0] (int): note (0-128)
    """
    # Count the frequency of each number in the array
    counter = Counter(arr)

    # Find the number with the highest frequency
    most_common = counter.most_common(1)

    # Return the most frequent number
    return most_common[0][0] if most_common else None



    
    
    
#####################################################################################
               #  Select a subset of octaves
#####################################################################################    
####### Below, keys to perform selection of two octaves if needed #################
    
C_key = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
Cs_key = [1, 13, 25, 37, 49, 61, 73, 85, 97, 109, 121]
D_key = [2, 14, 26, 38, 50, 62, 74, 86, 98, 110, 122]
Ds_key = [3, 15, 27, 39, 51, 63, 75, 87, 99, 111, 123]
E_key = [4, 16, 28, 40, 52, 64, 76, 88, 100, 112, 124]
F_key = [5, 17, 29, 41, 53, 65, 77, 89, 101, 113, 125]
Fs_key = [6, 18, 30, 42, 54, 66, 78, 90, 102, 114, 126]
G_key = [7, 19, 31, 43, 55, 67, 79, 91, 103, 115, 127]
Gs_key = [8, 20, 32, 44, 56, 68, 80, 92, 104, 116]
A_key = [9, 21, 33, 45, 57, 69, 81, 93, 105, 117]
As_key = [10, 22, 34, 46, 58, 70, 82, 94, 106, 118]
B_key = [11, 23, 35, 47, 59, 71, 83, 95, 107, 119]

#Selected octaves
C_restrict = [60,72]
Cs_restrict = [61,73]
D_restrict = [62, 74]
Ds_restrict = [63, 75]
E_restrict = [64, 76]
F_restrict = [65, 77]
Fs_restrict = [66, 78]
G_restrict = [67, 79]
Gs_restrict = [68, 80]
A_restrict = [69, 81]
As_restrict = [70, 82]
B_restrict = [71, 83]

def rescale(note):
    """
    Rescale note into the closest note of the same key belonging to the selected octaves
    
    Parameters:
    note (int): note in the (0-128) range

    Returns:
    note (int): note in the restricted octave
    """
    
    if note in C_key:
        note = min(C_restrict, key=lambda x: abs(x - note))
    elif note in Cs_key:
        note = min(Cs_restrict, key=lambda x: abs(x - note))
    elif note in D_key:
        note = min(D_restrict, key=lambda x: abs(x - note))
    elif note in Ds_key:
        note = min(Ds_restrict, key=lambda x: abs(x - note))
    elif note in E_key:
        note = min(E_restrict, key=lambda x: abs(x - note))
    elif note in F_key:
        note = min(F_restrict, key=lambda x: abs(x - note))
    elif note in Fs_key:
        note = min(Fs_restrict, key=lambda x: abs(x - note))
    elif note in G_key:
        note = min(G_restrict, key=lambda x: abs(x - note))
    elif note in Gs_key:
        note = min(Gs_restrict, key=lambda x: abs(x - note))
    elif note in A_key:
        note = min(A_restrict, key=lambda x: abs(x - note))
    elif note in As_key:
        note = min(As_restrict, key=lambda x: abs(x - note))
    elif note in B_key:
        note = min(B_restrict, key=lambda x: abs(x - note))
    
    return note
    
    
    
    
    
    
#####################################################################################
               #  Data augmentation strategies
#####################################################################################

def transpose(restrict, song, semitones):
    """
    Transpose melody of a number of semitones
    
    Parameters:
    restrict (bool): specifies if the input melody is restricted to two octaves or not;
                     If TRUE: all notes are transposed of one octave (up or down) so that they still
                     lie in the restricted octaves.
                     If FALSE: all notes are transposed of a random number of semitones (up to ± one octave)

    Returns:
    np.array(transposed): new transposed melody, shape [1,n_bars,npb,128]
    """
    
    if restrict:     
        transposed = []
        for i,note in enumerate(song):
            if note in C_restrict:
                note = max(C_restrict, key=lambda x: abs(x - note))
            elif note in Cs_restrict:
                note = max(Cs_restrict, key=lambda x: abs(x - note))
            elif note in D_restrict:
                note = max(D_restrict, key=lambda x: abs(x - note))
            elif note in Ds_restrict:
                note = max(Ds_restrict, key=lambda x: abs(x - note))
            elif note in E_restrict:
                note = max(E_restrict, key=lambda x: abs(x - note))
            elif note in F_restrict:
                note = max(F_restrict, key=lambda x: abs(x - note))
            elif note in Fs_restrict:
                note = max(Fs_restrict, key=lambda x: abs(x - note))
            elif note in G_key:
                note = max(G_restrict, key=lambda x: abs(x - note))
            elif note in Gs_restrict:
                note = max(Gs_restrict, key=lambda x: abs(x - note))
            elif note in A_key:
                note = max(A_restrict, key=lambda x: abs(x - note))
            elif note in As_restrict:
                note = max(As_restrict, key=lambda x: abs(x - note))
            elif note in B_restrict:
                note = max(B_restrict, key=lambda x: abs(x - note))

            transposed.append(note)
    else:
        transposed = [max(0, min(127, note + semitones)) for note in song]
        
    return(np.array(transposed))
    
    
    
    
def rotate(shift, song):
    """
    Rotate melody of a given number of places
    
    Parameters:
    shift_range (int): specifies range of possible number of places to shift the melody
    Returns:
    np.array(rotated): new rotated melody, shape [1,n_bars,npb,128]
    """
    # Apply the shift to the sequence
    rotated = np.concatenate((song[-shift:],song[:-shift]),axis=0)
    
    return(np.array(rotated))
    
    

def random_invert_song(song):

    # Circularly rotate the notes within the selected song along the axis of the bars
    inverted_song = np.flip(selected_song, axis=0)
    
    return inverted_song
    
    
    
    
def augmented_data(restrict, n_aug, song, n_bars, npb):
    """
    Augments dataset, adding variations to one data melody at a time
    
    Parameters:
    restrict (bool): specifies if the input melody is restricted to two octaves or not
    n_aug (int): specifies how many variations of the melody to create 
    song (ndarray): 2-D array of shape [n_bars x npb] sequence of notes 
    n_bars (int): number of bars per song
    npb (int): number of notes per bar

    Returns:
    augmented (ndarray): 3-D array of shape shape [1+n_aug,n_bars,npb] with original song + its variations
    """
    
    song = np.array(song)  #shape n_bars x npb
    augmented = song
    song = song.reshape(n_bars*npb,) #unwrap song into a list of notes, not divided by bars
    
    #number of rotation and transpositions
    n_transposed = int(n_aug/2)
    n_rotated = n_aug-n_transposed
    
    
    # Poissible semitones to transpose of
    semitones_list = list(range(-12,12))  # Transpose within ±12 semitones
    selected_semitones = random.sample(semitones_list, n_transposed)

    
    # Possible shifts for the rotation
    shifts_list = list(range(8, 120, 8))
    selected_shifts = random.sample(shifts_list, n_rotated)

    for i in range(n_transposed):
        semitones = selected_semitones[i]
        transposed = transpose(restrict, song, semitones).reshape(n_bars,npb)
        augmented = np.concatenate((augmented,transposed), axis=0)
        
    for j in range(n_rotated):
        shift = selected_shifts[j]
        rotated = rotate(shift, song).reshape(n_bars,npb)
        augmented = np.concatenate((augmented,rotated),axis=0)
    
    augmented = augmented.reshape(n_aug+1, n_bars, npb)
    
    # Add randomness: invert the whole song with a certain probability
    #r = random.random()
    #dim = 0
    #if r<p_inv:
    	#dim = 1
        #inverted = random_invert_song(song)
    	#augmented = np.concatenate((augmented,inverted),axis=0)
        
    #augmented = augmented.reshape(n_aug+1+dim, n_bars, npb)
        
    return(augmented)
    
    
    
    
    
########################################################################
             # One hot encoding + final formatting function
########################################################################

def one_hot_encoding(n_bars, npb, augmented, final_data):
    """
    Encodes each note (int in 0-128) into a one-hot encoded 1-D array of shape 128
    
    Parameters:
    n_bars (int): number of bars per song
    npb (int): number of notes per bar
    augmented (ndarray): 3-D array of shape shape [1+n_aug,n_bars,npb] with original song + its variations
    final_data (list): originally empty list; at each iteration, the new one-hot encoded version of each song
                       (4-D array of shape [1,n_bars,npb,128]) is appended.
    """
    
    
    for i in range(augmented.shape[0]):
        note_bars = np.zeros((int(n_bars),int(npb),128)) #array for storing one-hot enc. notes per song
        song = augmented[i,:,:]
        for j in range(augmented.shape[1]):
            bar = song[j,:]
            for k in range(augmented.shape[2]):
                note = bar[k]
                if note!=None :
                    note_bars[j,k,int(note)] = 1  #assign one to the corresponding note
        final_data.append(note_bars)
    
    
    
    
    
    
    
def formatting(midi_paths, tpb, tempo, npb, n_bars, restrict, shift_range, n_aug):
    """
    Starting from uncorrupted midi_files, for each midi:
    - Extracts and unpacks note events;
    - Divides note events into bars;
    - Divides bars into time-steps (depending on the number of npb chosen);
    - Augments dataset;
    - Formats note events into one-hot encoding array
    
    Parameters:
    midi_paths (list): list of paths of uncorrupted midi files
    tpb (int): ticks per beat, used to define length of a bar in ticks
    tempo (int): tempo, used to define length of a bar in ticks
    npb (int): notes per bar
    n_bars (int): number of bars per song
    restrict (bool): specifies if songs need to be restricted to two octaves
    shift_range (int): specifies range of possible number of places to shift the melody
    n_aug (int): specifies how many variations of the melody to create 

    Returns:
    np.array(final_data): 4-D array of shape [(len(midi_paths)*(n_aug+1)),n_bars,npb,128] of augmented formatted 
                          dataset
    """

    dataset = []

    # defined looking at the dataset
    bar_length = int(tpb*tempo)
    note_length = int(bar_length/npb)        #bar_length/16 to have the minimum timestep 1/16 of the bar
    nps = bar_length*n_bars                  #notes per song, to have 8 bars 

    
    # double check badly formatted files
    final_data = []
    # Process each MIDI file
    for iters,midi_file in enumerate(midi_paths):

        # open a midi file
        mid = MidiFile(midi_file)

        # extract notes and time in ticks of the notes
        notes,max_time = extract_note_events(midi_file, track_name='MELODY')
        notes = np.array(notes)

        # unpacks notes into a continuous time series representation.
        notes_unpacked = unpack_notes(notes, max_time)
        notes_unpacked = notes_unpacked[:nps] #cut each note list to obtain same size for everyone

        # create bars
        bars = []
        for j in range(0,(notes_unpacked.shape[0]-1),bar_length):
            bars.append(notes_unpacked[j:j+bar_length])            #shape 8xmax_time (notes x bars in ticks)
        

        n_bars = len(bars)
        note_bars = np.zeros((int(n_bars),int(npb),128))  #notes divided by bar, one-hot encoded, shape 8x16x128
        song_notes = []
        

        for i in range(0,len(bars)):
            bar = bars[i]
            bar_notes = []
            for t,j in enumerate(range(1,len(bar)-1,note_length)):
                timestep_bar = bar[j:j+note_length]
                note = most_frequent(timestep_bar)
                if restrict:
                   note = rescale(note)   #most frequent note in the timestep, ignoring silence, rescaled to 2 octaves
                bar_notes.append(note)   #notes in a bar (16)
                
             
            song_notes.append(bar_notes)  #notes in a song
            
                      
        ### DATA AUGMENTATION ###
        song_notes = np.array(song_notes)
        augmented = augmented_data(restrict, n_aug, song_notes, n_bars, npb)
        
        one_hot_encoding(n_bars, npb, augmented,final_data)

        if iters%500==0:
            print(iters,'/',len(midi_paths),' midis processed successfully.\n')
    return(np.array(final_data))

