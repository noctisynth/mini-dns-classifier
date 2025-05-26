import Header from '@/components/Header';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { useParams } from 'react-router';
import { Skeleton } from '@/components/ui/skeleton';
import { Label, Pie, PieChart } from 'recharts';
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from '@/components/ui/chart';
import { useEffect, useState } from 'react';
import * as api from '@/lib/api';
import { Badge } from '@/components/ui/badge';

export function Report() {
  const params = useParams<{ reportId: string }>();
  const reportId = params.reportId;
  if (!reportId) {
    throw new Error('Report ID is required');
  }

  const chartConfig = {
    packets: {
      label: 'Packets',
    },
  } satisfies ChartConfig;

  const [loaded, setLoaded] = useState<boolean>(false);
  const [reportData, setReportData] = useState<api.AnalysisReport | null>(null);
  const [createdAt, setCreatedAt] = useState<Date | null>(null);
  const pktChartData = [
    {
      class: 'Covert Channel',
      count: reportData?.covert_count || 0,
      fill: 'var(--chart-1)',
    },
    {
      class: 'Normal Traffic',
      count: reportData?.normal_count || 0,
      fill: 'var(--chart-2)',
    },
  ];
  const dnsChartData = [
    {
      class: 'Valid Packets',
      count: reportData?.valid_queries || 0,
      fill: 'var(--chart-3)',
    },
    {
      class: 'Invalid Packets',
      count:
        (reportData?.valid_queries &&
          reportData.total_packets - reportData.valid_queries) ||
        0,
      fill: 'var(--chart-1)',
    },
  ];
  // biome-ignore lint/correctness/useExhaustiveDependencies: only once
  useEffect(() => {
    api
      .getReport(reportId)
      .then((report) => {
        if (report.message) {
          console.log(report.message);
        }
        if (report.data) {
          setReportData(report.data.report);
          setCreatedAt(new Date(report.data.created_at));
        }
        console.log(report.data);
      })
      .finally(() => {
        setLoaded(true);
      });
  }, []);

  return (
    <div className="bg-background text-foreground h-screen w-screen">
      <Header />
      <main className="flex flex-col flex-1 w-full p-4">
        <div className="flex flex-col items-center justify-center">
          <h1 className="text-2xl font-bold mb-4">检查报告</h1>
          <p className="text-gray-600">报告ID: {params.reportId}</p>
        </div>
        <div className="flex flex-1 w-full">
          <Card className="w-full max-w-3xl mx-auto my-4 p-6">
            <h2 className="text-xl font-semibold mb-4">报告内容</h2>
            {!loaded && (
              <>
                <Skeleton className="w-[250px] h-[250px] rounded-full self-center" />
                <Skeleton className="w-full h-[30px] rounded-md" />
                <Skeleton className="w-full h-[30px] rounded-md" />
                <Skeleton className="w-[30%] h-[30px] rounded-md" />
              </>
            )}
            {loaded && (
              <>
                <header className="flex flex-col items-center">
                  <h2 className="text-xl font-semibold mb-1">数据包检测</h2>
                  <span className="text-gray-600">
                    创建时间: {createdAt?.toLocaleString()}
                  </span>
                </header>
                <ChartContainer
                  config={chartConfig}
                  className="aspect-square max-h-[250px]"
                >
                  <PieChart>
                    <ChartTooltip
                      cursor={false}
                      content={<ChartTooltipContent hideLabel />}
                    />
                    <Pie
                      data={dnsChartData}
                      dataKey="count"
                      nameKey="class"
                      innerRadius={60}
                      strokeWidth={5}
                    >
                      <Label
                        content={({ viewBox }) => {
                          if (viewBox && 'cx' in viewBox && 'cy' in viewBox) {
                            return (
                              <text
                                x={viewBox.cx}
                                y={viewBox.cy}
                                textAnchor="middle"
                                dominantBaseline="middle"
                              >
                                <tspan
                                  x={viewBox.cx}
                                  y={viewBox.cy}
                                  className="fill-foreground text-3xl font-bold"
                                >
                                  {reportData?.total_packets}
                                </tspan>
                                <tspan
                                  x={viewBox.cx}
                                  y={(viewBox.cy || 0) + 24}
                                  className="fill-muted-foreground"
                                >
                                  数据包
                                </tspan>
                              </text>
                            );
                          }
                        }}
                      />
                    </Pie>
                  </PieChart>
                </ChartContainer>
                <ChartContainer
                  config={chartConfig}
                  className="aspect-square max-h-[250px]"
                >
                  <PieChart>
                    <ChartTooltip
                      cursor={false}
                      content={<ChartTooltipContent hideLabel />}
                    />
                    <Pie
                      data={pktChartData}
                      dataKey="count"
                      nameKey="class"
                      innerRadius={60}
                      strokeWidth={5}
                    >
                      <Label
                        content={({ viewBox }) => {
                          if (viewBox && 'cx' in viewBox && 'cy' in viewBox) {
                            return (
                              <text
                                x={viewBox.cx}
                                y={viewBox.cy}
                                textAnchor="middle"
                                dominantBaseline="middle"
                              >
                                <tspan
                                  x={viewBox.cx}
                                  y={viewBox.cy}
                                  className="fill-foreground text-3xl font-bold"
                                >
                                  {reportData?.dns_packets}
                                </tspan>
                                <tspan
                                  x={viewBox.cx}
                                  y={(viewBox.cy || 0) + 24}
                                  className="fill-muted-foreground"
                                >
                                  DNS 数据包
                                </tspan>
                              </text>
                            );
                          }
                        }}
                      />
                    </Pie>
                  </PieChart>
                </ChartContainer>
                <h2 className="text-2xl font-bold">TOP5 恶意请求</h2>
                {(reportData?.top_covert_queries || []).map((predict, idx) => (
                  <Card key={`${idx}_${predict.query}`}>
                    <CardHeader>
                      <CardTitle className="h-[1.2rem] max-w-full overflow-x-hidden text-nowrap text-ellipsis">
                        请求：
                        <span className="text-blue-500">{predict.query}</span>
                      </CardTitle>
                      <CardDescription>
                        时间：{predict.timestamp}
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center justify-between">
                        <div>
                          疑似：
                          <span className="text-red-500">
                            {predict.prediction}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs">置信度</span>
                          <Badge
                            variant={
                              predict.confidence <= 0.8
                                ? 'default'
                                : 'destructive'
                            }
                          >
                            {predict.confidence <= 0.5
                              ? '较低'
                              : predict.confidence <= 0.8
                                ? '一般'
                                : predict.confidence <= 0.9
                                  ? '较高'
                                  : '非常高'}
                          </Badge>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </>
            )}
          </Card>
        </div>
      </main>
      <footer className="text-center text-sm text-gray-500 p-4">
        © 2023 DNS 分类系统
      </footer>
    </div>
  );
}
