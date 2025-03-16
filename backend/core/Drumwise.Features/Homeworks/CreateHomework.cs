using Drumwise.Application.Common.Errors;
using Drumwise.Application.Common.Models;
using Drumwise.Application.Entities;
using Drumwise.Features.Homeworks.Events;
using Drumwise.Infrastructure.Data;
using MediatR;
using Microsoft.AspNetCore.Http;

namespace Drumwise.Features.Homeworks;

public record CreateHomeworkCommand(string AssignedTo,
                                    string Title,
                                    string Exercise,
                                    DateTime Deadline) : IRequest<(Result, Guid)>;

public class CreateHomeworkHandler(ApplicationDbContext context, IHttpContextAccessor httpContextAccessor) 
    : IRequestHandler<CreateHomeworkCommand, (Result, Guid)>
{
    public async Task<(Result, Guid)> Handle(CreateHomeworkCommand request, CancellationToken cancellationToken)
    {
        // TODO: Implement adding attachments to homework (PDFs, videos, etc.) (add ResultNeeded property)
        var clientUrl = httpContextAccessor.HttpContext!.Items["ClientAddress"]!.ToString()!;
        
        var entity = new Homework()
        {
            AssignedTo = request.AssignedTo,
            HomeworkTitle = request.Title,
            Exercise = request.Exercise,
            Deadline = request.Deadline
        };
        
        entity.AddDomainEvent(new HomeworkCreatedEvent(entity, clientUrl));

        context.Homeworks.Add(entity);
        await context.SaveChangesAsync(cancellationToken).ConfigureAwait(false);

        return (Result.Success(ResultType.Ok), entity.Id);
    }
}
                                    