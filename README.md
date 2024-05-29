This is an industry project done for STS manufacturing, Hosur.
They are manufacturing a Brass bracket which is to be used in a defence vehicle which is manufactured by Ashok Leyland. 
The dimensions of the bracket is 80mm x 50mm. with a circle in the middle whose radius is 10mm. The company asked us to implement a vision based QA solution that checks the dimensions of the bracket based on the following conditions...

The tolerance was +or- 2%. 

We were also asked to check the location of the circle on the bracket and see if it is misplaced.

We had to implement this solution on a moving conveyor wherein we processed the video at almost realtime without stopping the conveyor.

To do this we have used a 4k 60fps camera and have fitted it over a conveyor belt in controlled lighting conditions.
The UI was developed using Streamlit.


We achieved a precision of 33 microns. (The uploaded file contains the PoC whose precision is 40 microns)


The link of the deployed app is
https://dimension-measurer.streamlit.app/
