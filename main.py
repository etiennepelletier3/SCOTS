import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt


params_path = "Results/Test 2/params.json"
# Load parameters from JSON file
with open(params_path, "r") as file:
    json_data = json.load(file)
m_rvec = np.array(json_data['mirror_rvec'])
m_rmtx = cv2.Rodrigues(m_rvec)[0]
m_tvec = np.array(json_data['mirror_tvec'])
s_rmtx = np.array(json_data['screen_rmtx'])
s_tvec = np.array(json_data['screen_tvec'])

# Plot 3D graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot([0, m_tvec[0][0]], [0, m_tvec[1][0]], [0, m_tvec[2][0]], c='k', linestyle='--')
ax.plot([0, s_tvec[0][0]], [0, s_tvec[1][0]], [0, s_tvec[2][0]], c='k', linestyle='--')
# Add tvec values to the plot
ax.text(m_tvec[0][0], m_tvec[1][0], m_tvec[2][0], f"({round(m_tvec[0][0], 4)}, {round(m_tvec[1][0], 4)}, {round(m_tvec[2][0], 4)})", color='r')
ax.text(s_tvec[0][0], s_tvec[1][0], s_tvec[2][0], f"({round(s_tvec[0][0], 4)}, {round(s_tvec[1][0], 4)}, {round(s_tvec[2][0], 4)})", color='b')

# Mirror position
ax.scatter(m_tvec[0], m_tvec[1], m_tvec[2], c='r', marker='o', label='Mirror')

# Screen position
ax.scatter(s_tvec[0], s_tvec[1], s_tvec[2], c='b', marker='o', label='Screen')

# Camera position
ax.scatter(0, 0, 0, c='g', marker='o', label='Camera')


# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Mirror and Screen Positions')

# Set legend
ax.legend()

# Show the plot
plt.show()
