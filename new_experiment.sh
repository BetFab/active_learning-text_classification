#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Illegal number of parameters - Only enter the name of the experiment"

else
	name=$1

	if [ -d "./experiments/"$name ]; then
		echo "This experiment already exists, if you continue the previous directory will be deleted"
		read -p "Are you sure ?" -n 1 -r
		echo 

		if [[ $REPLY =~ ^[Yy]$ ]]
		then
			# delete the directory
			echo "deleting  ... ./experiments/"$name
			rm -r "./experiments/"$name
		else
			exit 1
		fi
	fi	

	echo "creating ... ./experiments/"$name

	cp -r template_experiment/ experiments/$name
fi
