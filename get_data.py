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
    
    
    # Function to read MIDI file and extract note events in ticks
def extract_note_events(midi_file_path, track_name):
    midi_file = mido.MidiFile(midi_file_path)
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

    # If there is silence at the start
    #if notes[0, 0] > 0:
    #    notes_unpacked[0:int(notes[0, 0])] = 0
    
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
            # Note off: silence until the next note
            notes_unpacked[start_time:end_time] = 0

    # Handle the last note event, extending to the end of the time series
    last_time = int(notes[-1, 0])
    if notes[-1, 2] == 1:
        notes_unpacked[last_time:] = notes[-1, 1]
    else:
        notes_unpacked[last_time:] = 0

    return notes_unpacked



def most_frequent(arr):
    # Count the frequency of each number in the array
    counter = Counter(arr)

    # Find the number with the highest frequency
    most_common = counter.most_common(1)

    # Return the most frequent number
    return most_common[0][0] if most_common else None



def extract_melody_track(midi_file_path):
    midi = MidiFile(midi_file_path)

    for track in midi.tracks:
        if track.name=='MELODY':
            new_midi = track
    return new_midi
