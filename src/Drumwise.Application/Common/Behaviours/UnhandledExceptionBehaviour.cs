using MediatR;
using NLog;

namespace Drumwise.Application.Common.Behaviours;

public class UnhandledExceptionBehaviour<TRequest, TResponse>(ILogger logger) : IPipelineBehavior<TRequest, TResponse>
    where TRequest : notnull
{
    public async Task<TResponse> Handle(TRequest request, RequestHandlerDelegate<TResponse> next, CancellationToken cancellationToken)
    {
        try
        {
            return await next();
        }
        catch (Exception e)
        {
            const string requestName = nameof(TRequest);
            
            logger.Error(e, "Drumwise Request : Unhandled Exception for Request {Name}, {@Request}", 
                requestName, request);

            throw;
        }
    }
}