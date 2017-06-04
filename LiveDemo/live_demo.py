#! /usr/bin/env python
import sys
import subprocess

# needs "sudo apt-get install imagemagick"

if __name__ == "__main__":
	filename = 'input.png'
	print subprocess.check_output(['open', '-a', 'Photo Booth'])
	print subprocess.check_output(['imagesnap', '-w', '3', filename])
	print subprocess.check_output(['killall', 'Photo Booth'])
	print subprocess.check_output(['sips', '--resampleHeight', '76', filename])
	print subprocess.check_output(['convert', filename, '-gravity', 'Center', '-crop', '66x76+0+0', filename])
	print subprocess.check_output(['open', filename])