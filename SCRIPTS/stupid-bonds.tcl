# Clean up any previous installation
catch {
    rename stupidbonder_gui ""
    destroy .stupidbonder
    catch {menu tk delete StupidBonder}
    catch {vmd_uninstall_extension stupidbonder}
}

package provide stupidbonder 1.0

proc stupidbonder_gui {} {
    # Create the toplevel window
    if {[winfo exists .stupidbonder]} {
        wm deiconify .stupidbonder
        raise .stupidbonder
        return
    }
    
    toplevel .stupidbonder
    wm title .stupidbonder "StupidBonder"
    wm protocol .stupidbonder WM_DELETE_WINDOW { wm withdraw .stupidbonder }

    # Variables
    global selectedType processAllFrames nonPeriodicZ
    set selectedType 3
    set processAllFrames 0
    set nonPeriodicZ 0

    # Main frame
    frame .stupidbonder.main -padx 10 -pady 10
    pack .stupidbonder.main -fill both -expand 1

    # Atom type selection
    label .stupidbonder.main.label -text "Select atom type:"
    pack .stupidbonder.main.label -pady 5 -anchor w

    tk_optionMenu .stupidbonder.main.typeMenu selectedType 1 2 3 4 5
    pack .stupidbonder.main.typeMenu -pady 5 -fill x

    # Checkbuttons
    checkbutton .stupidbonder.main.allframes -text "Process all trajectory frames" \
        -variable processAllFrames -onvalue 1 -offvalue 0
    pack .stupidbonder.main.allframes -pady 5

    checkbutton .stupidbonder.main.nonperiodicz -text "Treat Z as non-periodic (ignore Z in bond checks)" \
        -variable nonPeriodicZ -onvalue 1 -offvalue 0
    pack .stupidbonder.main.nonperiodicz -pady 5

    # Action button
    button .stupidbonder.main.go -text "Remove Periodic Bonds" \
        -command {stupidbonder_go $selectedType $processAllFrames $nonPeriodicZ} \
        -width 25
    pack .stupidbonder.main.go -pady 10
    
    # Make sure window is visible
    wm deiconify .stupidbonder
    raise .stupidbonder
}

proc stupidbonder_go {typeValue allFrames nonPeriodicZ} {
    if {[molinfo num] == 0} {
        tk_messageBox -title "Error" -message "No molecule loaded!" -icon error
        return
    }

    # Get number of frames
    set num_frames [molinfo top get numframes]
    if {$num_frames == 0} {
        set num_frames 1
    }

    if {$allFrames} {
        set frame_range [seq 0 [expr {$num_frames - 1}]]
        puts -nonewline "\nProcessing $num_frames frames: "
        flush stdout
    } else {
        set frame_range [list [molinfo top get frame]]
        puts "\nProcessing current frame..."
    }
    
    puts "StupidBonder processing type $typeValue"
    if {$nonPeriodicZ} {
        puts "Treating Z-dimension as NON-PERIODIC (ignoring Z in bond length checks)"
    }

    # Select atoms of specified type
    set sel [atomselect top "type $typeValue"]
    set indices [$sel list]
    set num_atoms [$sel num]

    if {$num_atoms == 0} {
        tk_messageBox -title "Error" -message "No atoms of type $typeValue found!" -icon error
        $sel delete
        return
    }

    puts "Found $num_atoms atoms of type $typeValue"

    # Initialize counters
    set total_removed 0
    set total_kept 0
    set frames_with_removals 0

    # Process each frame
    foreach frame $frame_range {
        # Update frame
        animate goto $frame
        display update

        # Show progress
        if {$allFrames && [expr {$frame % 10 == 0}]} {
            puts -nonewline "."
            flush stdout
            if {[expr {$frame % 50 == 0}]} {
                puts ""
            }
        }

        # Get periodic box dimensions for this frame
        set box [molinfo top get {a b c alpha beta gamma} frame $frame]
        set a [lindex $box 0]
        set b [lindex $box 1]
        set c [lindex $box 2]

        if {$a <= 0 || $b <= 0 || $c <= 0} {
            set response [tk_messageBox -title "Warning" \
                -message "Box dimensions not set in frame $frame! Use default 100Å box?" \
                -icon warning -type yesno]
            
            if {$response eq "no"} {
                continue
            }
            set a 100; set b 100; set c 100
        }

        set half_a [expr {$a/2.0}]
        set half_b [expr {$b/2.0}]
        set half_c [expr {$c/2.0}]

        # Get coordinates for this frame
        $sel frame $frame
        set coords [$sel get {x y z}]
        set index_coord_map [dict create]
        foreach idx $indices coord $coords {
            dict set index_coord_map $idx $coord
        }

        # Get current bonds
        set all_bonds [topo getbondlist]
        set kept_bonds 0
        set removed_bonds 0
        set new_bondlist {}

        foreach bond $all_bonds {
            set a1 [lindex $bond 0]
            set a2 [lindex $bond 1]
            
            # Only process bonds between our selected atom type
            if {[dict exists $index_coord_map $a1] && [dict exists $index_coord_map $a2]} {
                set coord1 [dict get $index_coord_map $a1]
                set coord2 [dict get $index_coord_map $a2]
                
                # Calculate minimum image distance for X and Y
                set dx [expr {abs([lindex $coord1 0] - [lindex $coord2 0])}]
                set dy [expr {abs([lindex $coord1 1] - [lindex $coord2 1])}]
                set dz [expr {abs([lindex $coord1 2] - [lindex $coord2 2])}]

                # Check if bond is longer than allowed in XY (or XYZ if periodic)
                if {$dx > $half_a} {
                    incr removed_bonds
                } elseif {$dy > $half_b} {
                    incr removed_bonds
                } elseif {!$nonPeriodicZ && $dz > $half_c} {
                    # For non-periodic Z, completely ignore Z in distance calculation
                    incr removed_bonds
                } else {
                    lappend new_bondlist $bond
                    incr kept_bonds
                }
                
            } else {
                # Keep bonds not involving our selected atoms
                lappend new_bondlist $bond
                incr kept_bonds
            }
        }

        # Update bond list for this frame
        topo setbondlist $new_bondlist
        display update

        if {$removed_bonds > 0} {
            incr frames_with_removals
        }
        incr total_removed $removed_bonds
        incr total_kept $kept_bonds
    }

    # Clean up and notify user
    $sel delete

    if {$allFrames} {
        puts "\n\nProcessed all $num_frames frames"
        puts "Frames with bond removals: $frames_with_removals"
        puts "Total removed: $total_removed bonds"
        puts "Total kept: $total_kept bonds"
        puts "Average removed per frame: [format "%.1f" [expr {double($total_removed)/$num_frames}]]"
        
        set msg "Processed all $num_frames frames\n"
        append msg "Frames with removals: $frames_with_removals\n"
        append msg "Total removed: $total_removed bonds\n"
        append msg "Average: [format "%.1f" [expr {double($total_removed)/$num_frames}]] per frame"
    } else {
        set msg "Removed $total_removed bonds in current frame"
    }

    tk_messageBox -title "Done" -message $msg -icon info
}

# Helper proc for sequence generation
proc seq {from to} {
    set sequence {}
    for {set i $from} {$i <= $to} {incr i} {
        lappend sequence $i
    }
    return $sequence
}

# Try to register the extension
if {[catch {
    vmd_install_extension stupidbonder stupidbonder_gui "StupidBonder"
    puts "StupidBonder v1.0 successfully loaded"
    puts "Access via: Extensions → StupidBonder"
} err]} {
    puts "Note: Could not register menu item (normal for some VMD versions): $err"
    
    # Fallback option - create our own menu entry
    if {[catch {
        menu tk add StupidBonder command -label "StupidBonder" \
            -command stupidbonder_gui \
            -underline 0
        puts "Created alternative menu entry"
    } err2]} {
        puts "Could not create menu entry: $err2"
        puts "You can still run the GUI manually with: stupidbonder_gui"
    }
    
    # Ensure the window procedure exists
    proc ::tk_menuBar {} {}
}

# Auto-open the window when sourced
after idle stupidbonder_gui