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
    double[,] blendShapeWeights;

    public SkinnedMeshRenderer skinnedMeshRenderer;

    void Start()
    {
        matrixSize = blendShapesNum * sizeof(double); // size of 1x56 matrix of doubles
//        dataBuffer = new byte[matrixSize];
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
            dataBuffer = new byte[matrixSize];
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
                blendShapeWeights = new double[1, blendShapesNum];
                Buffer.BlockCopy(dataBuffer, 0, blendShapeWeights, 0, dataBuffer.Length);
//                Debug.Log("Received matrix:" + blendShapeWeights);

                UpdateBlendShapes();
            }
        }
    }

    private void UpdateBlendShapes()
        {
//             int n = blendShapeWeights.Length;
            for (int j = 0; j < blendShapesNum; j++)
            {
                skinnedMeshRenderer.SetBlendShapeWeight(j, 100*(float)blendShapeWeights[0, j]);
            }

//            for (int i = 0; i < blendShapeWeights.Length; i++)
//            {
//                int blendShapeIndex = indexMapping[i];
//                if (blendShapeIndex < skinnedMeshRenderer.sharedMesh.blendShapeCount)
//                {
//                    skinnedMeshRenderer.SetBlendShapeWeight(blendShapeIndex, 100*blendShapeWeights[i]);
//                }
//                else
//                {
//                    Debug.LogWarning("Blend shape index " + blendShapeIndex + " is out of range!");
//                }
//            }
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
