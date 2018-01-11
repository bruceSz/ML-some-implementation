/***************************************************************************
 * 
 * Copyright (c) 2017 jd.com, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file CNeuralNet.h
 * @author brucesz(zhangsong5@jd.com)
 * @date 2017/04/15 19:29:35
 * @version $Revision$ 
 * @brief 
 *  
 **/
#ifndef CNEURALNET_H
#define CNEURALNET_H

#endif  // CNEURALNET_H

/* vim: set ts=4 sw=4 sts=4 tw=100 */
#inlcude <vector>

struct SNeutron
{
    // number of inputs of neutron cell
    int m_NumInputs;

    // weight for each input
    vector<double> m_vecWeight;

    SNeutron(int NumInputs);
};

struct SNeutronLayer
{
    int m_NumNeutrons;

    vector<SNeutron> m_vecNeutrons;
    SNeutronLayer(int NumNeutrons, int NumInputsPerNeutron);
};

class CNeuralNet
{
   private:
    int m_NumInputs;
    int m_NumOutputs;
    int m_NumHiddenLayers;
    int m_NeutronsPerHiddenLyr;
    vector<SNeutronLayer> m_vecLayers;
   public:

    CNeuralNet();
    void CreateNet();
    vector<double> GetWeights() const;
    int GetNumberOfWeights() const;
    void PutWeights(vector<double> &weights);
    inline double Sigmoid(double activation, double response);
    vector<double> Update(vector<double> &inputs);
};
