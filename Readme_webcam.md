In order to run the enable the webcam in WSL you need to do the following:
1. Connect the webcam to the host machine
2. Run the following command in powershell with admin privelages: 
```bash
winget install --interactive --exact dorssel.usbipd-win
usbipd list
```
3. From here, write down the BUSID of the webcam you want to use.
4. Run the following command in powershell with admin privelages: 
```bash
usbipd bind --busid <BUSID>
```
For example in our case the BUSID was 1-1
Then run 
```bash
usbipd attach --wsl --busid <BUSID>
```
5. Run the following command in WSL: 
```bash
sudo usbipd attach --busid <BUSID>
```
6. Run the following command in WSL: 
```bash
lsusb
```
7. You should see the webcam in the list.
Check if the video device node was created:
```bash
ls /dev/video*
```
8. Run the following command in WSL: 
```bash
python3 Demo.py
```