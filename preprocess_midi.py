import mido
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import pretty_midi
from collections import Counter
from mido import MidiFile, MidiTrack, Message

def search_melodies(path):
    
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
    
    
    
#Function to select tracks with specific ticks per beat and tempo
def select_tempo(tpb, tempo_num, tempo_den):
    # select 384 ticks per beat and 4/4 tempo
    indexes_ = []
    selected_meta = []
    selected_paths = []
    for i in range(len(metadata)):
        if metadata[i][0] == tpb:
            for msg in metadata[i][1]:
                if msg.type == 'time_signature' and msg.numerator == tempo_num and msg.denominator == tempo_den:
                    indexes_.append(i)
                    selected_paths.append(paths_new[i])
                    selected_meta.append(metadata[i])
                    break  # Exit the inner loop once a matching time_signature is found
    return(indexes_, selected_paths, selected_meta)
    
    
    # Function to read MIDI file and extract note events in ticks
def extract_note_events(path, track_name):
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
    Unpacks note events into a continuous time series representation.

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
    # Count the frequency of each number in the array
    counter = Counter(arr)

    # Find the number with the highest frequency
    most_common = counter.most_common(1)

    # Return the most frequent number
    return most_common[0][0] if most_common else None



def extract_melody_track(path):
    midi = MidiFile(path)

    for track in midi.tracks:
        if track.name=='MELODY':
            new_midi = track
    return new_midi
    
    
    
    
    
    
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
    
    
    
    
    
    
    
def formatting(midi_paths, tpb, tempo, npb, n_bars):

    dataset = []

    # defined looking at the dataset
    bar_length = int(tpb*tempo)
    note_length = int(bar_length/npb)        #bar_length/16 to have the minimum timestep 1/16 of the bar
    nps = bar_length*n_bars                  #notes per song, to have 8 bars 

    
    # double check badly formatted files
    icount = 0 

    # Process each MIDI file
    for iters,midi_file in enumerate(midi_paths):

        # open a midi file
        mid = MidiFile(midi_file)

        # extract notes and time in ticks of the notes
        notes,max_time = extract_note_events(midi_file, track_name='MELODY')
        notes = np.array(notes)

        if (notes.shape[0]==0):
            icount +=1
            continue

        # unpacks notes into a continuous time series representation.
        notes_unpacked = unpack_notes(notes, max_time)
        notes_unpacked = notes_unpacked[:nps] #cut each note list to obtain same size for everyone

        # create bars
        bars = []
        for j in range(0,(notes_unpacked.shape[0]-1),bar_length):
            bars.append(notes_unpacked[j:j+bar_length])

        n_bars = len(bars)
        note_bars = np.zeros((int(n_bars),int(npb),128))  #notes divided by bar, shape 8x16x128
        note_bars_list = []

        for i in range(0,len(bars)):
            bar = bars[i]
            notes_list = []
            for t,j in enumerate(range(1,len(bar)-1,note_length)):
                timestep_bar = bar[j:j+note_length]
                note = rescale(most_frequent(timestep_bar))   #most frequent note in the timestep, ignoring silence, rescaled to 2 octaves
                notes_list.append(note)

                if note!=None :
                    note_bars[i,t,int(note)] = 1  #assign one to the corresponding note
            note_bars_list.append(notes_list)

        dataset.append(note_bars)
        if iters%500==0:
            print(iters,'/',len(midi_files),' midis processed successfully.\n')
    return dataset

