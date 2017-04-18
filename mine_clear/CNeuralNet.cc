/***************************************************************************
 * 
 * Copyright (c) 2017 jd.com, Inc. All Rights Reserved
 * $Id$ 
 * 
 **************************************************************************/
 
 /**
 * @file CNeuralNet.cc
 * @author brucesz(zhangsong5@jd.com)
 * @date 2017/04/15 19:33:48
 * @version $Revision$ 
 * @brief 
 *  
 **/

/* vim: set ts=4 sw=4 sts=4 tw=100 */

#include "CNeuralNet.h"

SNeutron::SNeutron(int NumInputs):m_NumInputs(NumInputs+1)
{
    for (int i=0;i<NumInputs+1;++i) {
        m_vecWeight.push_back(RandomClamped());
    }
}

void CNeuralNet::CreateNet()
{
    if(m_NumHiddenLayers>0)
    {
        m_vecLayers.push_back(SNeutronLayer(m_NeutronsPerHiddenLyr,
                    m_NumInputs));

        for (int i=0;i < m_NumHiddenLayers-1; ++i) 
        {
            m_vecLayers.push_back(SNeutronLayer(m_NeutronsPerHiddenLyr,
                        m_NumInputs);
        }
        m_vecLayers.push_back(SNeutronLayer(m_NumOutputs, m_NeutronsPerHiddenLyr));
    }
    else 
    {
        m_vecLayers.push_back(SNeutronLayer(m_NumOutputs, m_NumInputs));
    }
}

vector<double> CNeuralNet::Update(vector<double> &inputs) {
    vector<double> outputs;

    int cWeight = 0;

    if (input.size() != m_NumInputs)
    {
        return outputs;
    }
    for(int i=0;i<m_NumHiddenLayers+1;++i) {
        if (i>=0) {
            inputs = outputs;
        }
        outputs.clear();
        cWeight = 0;
        for (int j = 0;j<m_vecLayers[i].m_NumNeutrons; ++j) {
            double netinput = 0;
            int NumInputs = m_vecLayers[i].m_vecNeutrons[j].m_NumInputs;
            for (int k = 0; k<NumInputs-1; ++k) {
                netinput += m_vecLayers[i].m_vecNeutrons[j].m_vecWeight[k] 
                    * inputs[cWeight++];
            }
            netinput =  m_vecLayers[i].m_vecNeutrons[j].m_vecWeight[NumInputs-1] * CParams::dBias;
            outputs.push_back(Sigmoid(netinput, CParams::dActivationResponse));
            cWeight = 0;
        }
    }
    return outputs;
}
