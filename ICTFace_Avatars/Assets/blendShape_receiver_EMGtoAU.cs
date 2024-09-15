using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Net.Sockets;
using System.Text;

public class blendShape_receiver_EMGtoAU : MonoBehaviour
{
    TcpClient client;
    NetworkStream stream;
    byte[] dataBuffer;
    private const int blendShapesNum = 57;

    int matrixSize;
    float[] blendShapeWeights; // Change to float

    public SkinnedMeshRenderer skinnedMeshRenderer;

    void Start()
    {
        matrixSize = blendShapesNum * sizeof(float); // size of 1x57 matrix of floats
        dataBuffer = new byte[matrixSize]; // Initialize dataBuffer
        ConnectToServer();

        // Get the SkinnedMeshRenderer component
        if (!skinnedMeshRenderer)
            skinnedMeshRenderer = GetComponent<SkinnedMeshRenderer>();
    }

    void ConnectToServer()
    {
        try
        {
            client = new TcpClient("localhost", 65432);
            stream = client.GetStream();
            Debug.Log("Connected to server");
        }
        catch (Exception e)
        {
            Debug.Log("Socket error: " + e);
        }
    }

    void Update()
    {
        if (stream != null && stream.DataAvailable)
        {
            int bytesRead = stream.Read(dataBuffer, 0, dataBuffer.Length);
            if (bytesRead == matrixSize)
            {
                blendShapeWeights = new float[blendShapesNum]; // Change to float array
                Buffer.BlockCopy(dataBuffer, 0, blendShapeWeights, 0, dataBuffer.Length);

                UpdateBlendShapes();
            }
        }
    }

    private void UpdateBlendShapes()
    {
        for (int j = 0; j < blendShapesNum; j++)
        {
            // Clamp the values between 0 and 100
            float weight = Mathf.Clamp(blendShapeWeights[j] * 100, 0, 100);
            skinnedMeshRenderer.SetBlendShapeWeight(j, weight);
        }
    }

    void OnApplicationQuit()
    {
        if (stream != null)
        {
            stream.Close();
        }
        if (client != null)
        {
            client.Close();
        }
    }
}